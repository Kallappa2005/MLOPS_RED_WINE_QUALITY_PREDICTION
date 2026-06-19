"""
Wine Quality Prediction - Flask Application

Security hardening:
  - /train requires a secret token via TRAIN_SECRET env var (fail-closed if unset)
  - threading.Lock() replaces the bare boolean race condition
  - Flask-Limiter caps /train at 10 req/hour per IP, /train/status at 60/min
  - /train/status also requires the same token
  - /models/* endpoints require ADMIN_TOKEN (settable via env var or X-Admin-Token header)
  - /models/rollback is rate-limited to 10 req/hour per IP
  - /models/list, /models/compare, /models/<version_id> are rate-limited to 30 req/min
  - debug mode is controlled by FLASK_DEBUG env var, defaults to off in production
  - All admin actions are logged with caller IP and details

Additional improvements:
  - /health endpoint for uptime monitoring
  - training_log capped at MAX_LOG_LINES to prevent unbounded memory growth
"""

import functools
import json
import os
import secrets
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
import numpy as np
import pandas as pd
import portalocker
from flask import Flask, abort, jsonify, render_template, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pathlib import Path
from mlProject.config.configuration import ConfigurationManager
from mlProject.constants import ENV_FLASK_PORT, ENV_FLASK_DEBUG, ENV_TAG
from mlProject.pipeline.prediction import PredictionPipeline
from mlProject import logger
from mlProject.utils.common import load_env_file, get_env_or_config
from mlProject.utils.model_registry import load_registry, rollback_to_version
from mlProject.utils.unified_registry import get_unified_registry


def _get_registry_path() -> Path:
    """Get the configured model registry path."""
    try:
        return ConfigurationManager().get_model_registry_config().registry_path
    except Exception:
        return Path("artifacts/model_registry.json")

load_env_file()

app = Flask(__name__)

# Global pipeline instance — loaded once at startup to avoid per-request disk I/O
pipeline = PredictionPipeline()

# ---------------------------------------------------------------------------
# Rate limiter - keyed on caller's IP
# ---------------------------------------------------------------------------
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[],          # No blanket global limit; set per-route
    storage_uri="memory://",    # Fine for single-worker Render free tier
)

# ---------------------------------------------------------------------------
# Training state
# _training_lock is held for the entire duration of a training run.
# acquire(blocking=False) is atomic - no race condition possible.
#
# Memory & thread safety:
# - training_log is a bounded deque(maxlen=100) to prevent unbounded growth
# - _log_lock protects all reads/writes to training_log
# - ThreadPoolExecutor(max_workers=1) replaces raw thread creation
# - Timeout mechanism auto-releases lock if training exceeds TRAIN_TIMEOUT
# - _training_file_lock_path provides cross-process exclusion for DVC/CLI path
# ---------------------------------------------------------------------------
_training_lock = threading.Lock()
_log_lock = threading.Lock()
is_training = False
MAX_LOG_LINES = 100
training_log = deque(maxlen=MAX_LOG_LINES)
_train_executor = ThreadPoolExecutor(max_workers=1)
_training_process = None
_training_process_lock = threading.Lock()
TRAIN_TIMEOUT = int(os.environ.get("TRAIN_TIMEOUT", "1800"))  # default 30 min

_TRAINING_LOCK_FILE = Path("artifacts/.train.lock")
_training_file_lock_fd = None
_training_file_lock_lock = threading.Lock()


def _acquire_training_file_lock() -> bool:
    """Acquire a cross-process file lock for training. Returns True if acquired."""
    global _training_file_lock_fd
    with _training_file_lock_lock:
        if _training_file_lock_fd is not None:
            return False
        _TRAINING_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = open(_TRAINING_LOCK_FILE, "w")
            portalocker.lock(fd, portalocker.LOCK_EX | portalocker.LOCK_NB)
            _training_file_lock_fd = fd
            return True
        except (IOError, portalocker.LockException):
            return False


def _release_training_file_lock():
    """Release the cross-process training file lock."""
    global _training_file_lock_fd
    with _training_file_lock_lock:
        if _training_file_lock_fd is not None:
            try:
                portalocker.unlock(_training_file_lock_fd)
                _training_file_lock_fd.close()
            except Exception:
                pass
            _training_file_lock_fd = None



# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------
def _verify_train_token() -> bool:
    """
    Validate the caller's secret token against the TRAIN_SECRET env var.

    Accepts the token in two ways (works from browser tab AND curl/CI):
      - Query param  : GET /train?token=<secret>
      - Custom header: X-Train-Token: <secret>

    Returns False (deny) if TRAIN_SECRET is not set in the environment -
    the endpoint is disabled entirely by default (fail-closed).
    Uses secrets.compare_digest to prevent timing-oracle attacks.
    """
    expected = os.environ.get("TRAIN_SECRET", "")
    if not expected:
        return False  # No secret configured - refuse everything

    supplied = (
        request.args.get("token", "")
        or request.headers.get("X-Train-Token", "")
    )
    return secrets.compare_digest(supplied.encode(), expected.encode())


def get_current_production_version(registry_path):
    """Get the current production version ID from the registry."""
    registry = load_registry(registry_path)
    return registry.get("production")


def log_admin_action(action, details=""):
    """Log an admin action with caller IP and details."""
    app.logger.warning(
        f"Admin action: {action}, "
        f"caller={request.remote_addr}, "
        f"details={details}"
    )


def require_admin_token(f):
    """Decorator to require admin token for model management operations."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        token = (request.headers.get('X-Admin-Token')
                 or request.headers.get('X-Train-Token')
                 or request.args.get("token", ""))
        expected = os.environ.get("ADMIN_TOKEN") or os.environ.get("TRAIN_SECRET", "")
        if not expected:
            return jsonify({"error": "Admin token not configured on server"}), 500
        if not token or not secrets.compare_digest(token.encode(), expected.encode()):
            return jsonify({"error": "Invalid or missing admin token"}), 401
        log_admin_action(f.__name__)
        return f(*args, **kwargs)
    return decorated


# ---------------------------------------------------------------------------
# Log helper
# ---------------------------------------------------------------------------
def _safe_log(msg: str) -> None:
    """Append a message to training_log, capped at MAX_LOG_LINES."""
    if len(training_log) < MAX_LOG_LINES:
        training_log.append(msg)


# ---------------------------------------------------------------------------
# Startup helper
# ---------------------------------------------------------------------------
def validate_config_at_startup() -> None:
    """Validate all required configs are present before starting the server."""
    required_configs = [
        ("config/config.yaml", "Main configuration file"),
        ("params.yaml", "Parameters file"),
        ("schema.yaml", "Schema file"),
    ]
    missing = []
    for path, desc in required_configs:
        if not Path(path).exists():
            missing.append(f"{desc} ({path})")
    if missing:
        print(f"ERROR: Missing required configuration files: {', '.join(missing)}")
        sys.exit(1)
    print(f"Configuration validation passed. Environment: {os.environ.get(ENV_TAG, 'development')}")


# ---------------------------------------------------------------------------
# Background training worker
# ---------------------------------------------------------------------------
def _run_training_in_background() -> None:
    """Subprocess-based training; releases _training_lock when done."""
    global is_training, _training_process
    if not _acquire_training_file_lock():
        with _log_lock:
            training_log.append("Training rejected: another process is already training")
        try:
            _training_lock.release()
        except RuntimeError:
            pass
        return
    start_time = time.time()
    try:
        with _log_lock:
            training_log.append("Training started...")
        proc = subprocess.Popen(
            [sys.executable, "main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        with _training_process_lock:
            _training_process = proc
        try:
            stdout, stderr = proc.communicate(timeout=TRAIN_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            raise
        with _log_lock:
            if proc.returncode == 0:
                training_log.append("Training completed successfully!")
                training_log.append(stdout)
            else:
                training_log.append("Training failed!")
                training_log.append(stderr or stdout)
    except subprocess.TimeoutExpired:
        with _log_lock:
            training_log.append(
                f"Training timed out after {TRAIN_TIMEOUT}s"
            )
    except Exception as exc:
        with _log_lock:
            training_log.append(f"Training error: {exc}")
    finally:
        is_training = False
        _release_training_file_lock()
        with _training_process_lock:
            _training_process = None
        try:
            _training_lock.release()
        except RuntimeError:
            pass  # already released - should never happen


def ensure_model_trained() -> None:
    """Train the model automatically on first deploy if no artifact exists."""
    try:
        config_manager = ConfigurationManager()
        eval_config = config_manager.get_model_evaluation_config()
        model_path = eval_config.model_path
    except Exception:
        model_path = Path("artifacts/model_trainer/model.joblib")

    if not model_path.exists():
        if not _acquire_training_file_lock():
            print("Auto-training skipped: another process is already training")
            return
        print("Model not found - starting automatic training...")
        try:
            result = subprocess.run(
                [sys.executable, "main.py"],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                print("Auto-training completed!")
            else:
                print(f"Auto-training failed:\n{result.stderr}")
        except Exception as exc:
            print(f"Auto-training failed: {exc}")
        finally:
            _release_training_file_lock()
    else:
        from mlProject.utils.common import verify_model_integrity
        from mlProject.utils.model_registry import load_registry
        checksum_path = Path(str(model_path) + ".sha256")
        if not verify_model_integrity(model_path, checksum_path):
            print("Model integrity check FAILED - consider retraining.")
        else:
            print("Model already exists - ready for predictions!")
        # Check if active model matches registry production version
        try:
            registry_path = _get_registry_path()
            registry = load_registry(registry_path)
            prod_id = registry.get("production")
            model_info_path = model_path.parent / "model_info.json"
            if prod_id and model_info_path.exists():
                with open(model_info_path) as f:
                    model_info = json.load(f)
                loaded_version = model_info.get("version_id")
                if loaded_version and loaded_version != prod_id:
                    print(
                        f"WARNING: Active model version {loaded_version} does not match "
                        f"registry production version {prod_id}."
                    )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def homePage():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """Lightweight liveness probe for uptime monitors and load balancers."""
    health_data = {"status": "ok", "is_training": is_training}
    try:
        registry_path = _get_registry_path()
        registry = load_registry(registry_path)
        prod_id = registry.get("production")
        health_data["production_version"] = prod_id
        model_path = Path("artifacts/model_trainer/model.joblib")
        if model_path.exists():
            model_info_path = model_path.parent / "model_info.json"
            if model_info_path.exists():
                with open(model_info_path) as f:
                    model_info = json.load(f)
                loaded_version = model_info.get("version_id")
                health_data["active_model_version"] = loaded_version
                if prod_id and loaded_version and loaded_version != prod_id:
                    health_data["version_mismatch"] = True
    except Exception:
        pass
    return jsonify(health_data), 200


@app.route("/health/registry", methods=["GET"])
def registry_health():
    """Health check endpoint to verify registry consistency between JSON and MLflow."""
    try:
        registry_path = _get_registry_path()
        unified_registry = get_unified_registry(registry_path)
        issues = unified_registry.validate()

        health_status = {
            "status": "healthy" if not issues else "degraded",
            "issues_count": len(issues),
            "issues": issues,
        }

        # Add sync status if MLflow is enabled
        if unified_registry.mlflow_tracker and unified_registry.mlflow_tracker.use_mlflow:
            health_status["mlflow_enabled"] = True
        else:
            health_status["mlflow_enabled"] = False

        status_code = 200 if not issues else 200  # Still return 200 for monitoring
        return jsonify(health_status), status_code
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Registry health check failed: {str(e)}",
        }), 500


@app.route("/health/registry/sync", methods=["POST"])
@require_admin_token
def sync_registries():
    """Sync MLflow registry state from JSON registry."""
    try:
        registry_path = _get_registry_path()
        unified_registry = get_unified_registry(registry_path)
        result = unified_registry.sync_from_json()

        log_admin_action(
            "registry_sync",
            f"synced={result['synced']}, actions={len(result['actions'])}"
        )

        return jsonify({
            "message": "Registry sync completed",
            "synced": result["synced"],
            "actions": result["actions"],
        })
    except Exception as e:
        return jsonify({
            "error": f"Registry sync failed: {str(e)}",
        }), 500


@app.route("/train", methods=["GET"])
@limiter.limit("10 per hour")           # Hard cap: 10 triggers/hr per IP
def training():
    global is_training, training_log

    # 1 - Authenticate first (401 reveals nothing about current state)
    if not _verify_train_token():
        abort(401)

    # 2 - Atomically acquire the lock; reject if already running
    acquired = _training_lock.acquire(blocking=False)
    if not acquired:
        return render_template(
            "train_status.html",
            training_success=None,
            training_log="Training is already in progress! Please wait...",
        )

    # 3 - Mark running and reset log buffer under lock
    is_training = True
    with _log_lock:
        training_log.clear()

    # 4 - Submit to ThreadPoolExecutor (reuses worker thread, avoids thread leak)
    _train_executor.submit(_run_training_in_background)

    return render_template(
        "train_status.html",
        training_success=True,
        training_log="Training started in background! Check /train/status for updates.",
    )


@app.route("/train/status", methods=["GET"])
@limiter.limit("60 per minute")         # Polling endpoint - more generous
def training_status():
    """Return current training state. Also requires the token."""
    if not _verify_train_token():
        abort(401)

    with _log_lock:
        log_snapshot = list(training_log)


    return jsonify({
        "is_training": is_training,
        "log": log_snapshot,
    })


@app.route("/predict", methods=["POST", "GET"])
def index():
    if request.method == "POST":
        try:
            fixed_acidity        = float(request.form["fixed_acidity"])
            volatile_acidity     = float(request.form["volatile_acidity"])
            citric_acid          = float(request.form["citric_acid"])
            residual_sugar       = float(request.form["residual_sugar"])
            chlorides            = float(request.form["chlorides"])
            free_sulfur_dioxide  = float(request.form["free_sulfur_dioxide"])
            total_sulfur_dioxide = float(request.form["total_sulfur_dioxide"])
            density              = float(request.form["density"])
            pH                   = float(request.form["pH"])
            sulphates            = float(request.form["sulphates"])
            alcohol              = float(request.form["alcohol"])

            # Boundary validation checks
            if fixed_acidity <= 0:
                raise ValueError("Fixed Acidity must be positive.")
            if volatile_acidity <= 0:
                raise ValueError("Volatile Acidity must be positive.")
            if citric_acid < 0:
                raise ValueError("Citric Acid must be non-negative.")
            if residual_sugar <= 0:
                raise ValueError("Residual Sugar must be positive.")
            if chlorides < 0:
                raise ValueError("Chlorides must be non-negative.")
            if free_sulfur_dioxide < 0:
                raise ValueError("Free Sulfur Dioxide must be non-negative.")
            if total_sulfur_dioxide < 0:
                raise ValueError("Total Sulfur Dioxide must be non-negative.")
            if density <= 0:
                raise ValueError("Density must be positive.")
            if not (0 < pH < 14):
                raise ValueError("pH must be between 0 and 14.")
            if sulphates < 0:
                raise ValueError("Sulphates must be non-negative.")
            if alcohol <= 0:
                raise ValueError("Alcohol must be positive.")

            data = np.array([
                fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                density, pH, sulphates, alcohol,
            ]).reshape(1, 11)

            predict = pipeline.predict(data)
            final_prediction = round(float(predict[0]), 2)

            return render_template("results.html", prediction=final_prediction)

        except ValueError as exc:
            logger.error(f"Validation error in /predict: {exc}")
            return render_template(
                "results.html",
                error_msg=f"Validation error: {exc}",
            ), 400
        except Exception as exc:
            logger.error(f"Unexpected error in /predict: {exc}")
            return render_template(
                "results.html",
                error_msg="An unexpected error occurred. Please try again.",
            ), 500
    else:
        return render_template("index.html")


@app.route('/models', methods=['GET'])
@limiter.limit("30 per minute")
@require_admin_token
def list_models():
    """List all registered model versions."""
    registry_path = _get_registry_path()
    registry = load_registry(registry_path)
    log_admin_action("list_models", f"versions_count={len(registry.get('versions', []))}")
    return jsonify(registry)


@app.route('/models/compare', methods=['GET'])
@limiter.limit("30 per minute")
@require_admin_token
def compare_models():
    """Show metric diff between current and previous model."""
    comparison_path = Path('artifacts/model_evaluation/metrics_comparison.json')
    if comparison_path.exists():
        try:
            with open(comparison_path) as f:
                comparison = json.load(f)
            log_admin_action("compare_models", "comparison_data_returned")
            return jsonify(comparison)
        except Exception as e:
            log_admin_action("compare_models", f"read_error={str(e)}")
            return jsonify({"error": str(e)}), 500
    log_admin_action("compare_models", "no_comparison_data")
    return jsonify({"message": "No comparison data available"})


@app.route('/models/rollback', methods=['POST'])
@limiter.limit("10 per hour")
@require_admin_token
def rollback_model():
    """Rollback production alias to a specified version and restore the model file."""
    version_id = request.json.get("version_id")
    if not version_id:
        return jsonify({"error": "version_id is required"}), 400
    registry_path = _get_registry_path()
    current_prod = get_current_production_version(registry_path)
    log_admin_action(
        "rollback_initiated",
        f"target_version={version_id}, current_production={current_prod}"
    )
    unified_registry = get_unified_registry(registry_path)
    if unified_registry.rollback_to_version(version_id):
        log_admin_action("rollback_success", f"rolled_back_to={version_id}")
        return jsonify({"message": f"Rolled back to version {version_id} and restored model file"})
    log_admin_action("rollback_failed", f"version_{version_id}_not_found")
    return jsonify({"error": f"Rollback failed: version {version_id} not found or model file missing"}), 404


@app.route('/models/<version_id>', methods=['GET'])
@limiter.limit("30 per minute")
@require_admin_token
def get_model_version(version_id):
    """View metadata for a specific version."""
    registry_path = _get_registry_path()
    registry = load_registry(registry_path)
    for v in registry.get("versions", []):
        if v.get("id") == version_id:
            log_admin_action("get_model_version", f"version_id={version_id}")
            return jsonify(v)
    log_admin_action("get_model_version", f"version_{version_id}_not_found")
    return jsonify({"error": f"Version {version_id} not found"}), 404


@app.route('/mlflow', methods=['GET'])
def mlflow_ui():
    """Redirect to MLflow Tracking UI if configured."""
    try:
        config_manager = ConfigurationManager()
        registry_config = config_manager.get_model_registry_config()
        if not registry_config.use_mlflow:
            return jsonify({
                "message": "MLflow is not enabled. Set use_mlflow: true in config.yaml or ENV_MLFLOW_USE_MLFLOW=true.",
                "enabled": False,
            })
        tracking_uri = registry_config.mlflow_tracking_uri
        parsed = urlparse(tracking_uri)
        is_local = parsed.scheme == "" or parsed.scheme == "file"
        if is_local:
            return jsonify({
                "message": "MLflow tracking URI is local. Run 'mlflow ui' in your terminal, then visit http://localhost:5000.",
                "enabled": True,
                "tracking_uri": tracking_uri,
                "mlflow_ui_command": "mlflow ui",
                "local_ui_url": "http://localhost:5000",
                "experiment_name": registry_config.mlflow_experiment_name,
                "model_name": registry_config.mlflow_model_name,
            })
        return jsonify({
            "message": "MLflow Tracking Server configured",
            "enabled": True,
            "tracking_uri": tracking_uri,
            "experiment_name": registry_config.mlflow_experiment_name,
            "model_name": registry_config.mlflow_model_name,
        })
    except Exception as e:
        return jsonify({
            "message": f"Failed to read MLflow configuration: {str(e)}",
            "enabled": False,
        }), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _shutdown_handler(signum, frame):
    """Clean up training subprocess on shutdown signals."""
    print(f"Received signal {signum}, shutting down...")
    global _training_process
    with _training_process_lock:
        if _training_process is not None:
            print("Terminating training subprocess...")
            _training_process.terminate()
    sys.exit(0)


# ---------------------------------------------------------------------------
# Startup signal handlers — runs at import time so Gunicorn workers inherit.
# ---------------------------------------------------------------------------
signal.signal(signal.SIGTERM, _shutdown_handler)
signal.signal(signal.SIGINT, _shutdown_handler)
validate_config_at_startup()


if __name__ == "__main__":
    # Local development server (not used in Docker/Gunicorn production).
    # debug=True in production exposes an interactive shell - never do this.
    # Set FLASK_DEBUG=1 locally to enable the Werkzeug debugger.
    port = int(get_env_or_config(ENV_FLASK_PORT, "8080", transform=int))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"

    app.run(host="0.0.0.0", port=port, debug=debug)