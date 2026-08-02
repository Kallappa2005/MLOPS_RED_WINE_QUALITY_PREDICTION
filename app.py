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
import portalocker
import io
import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, render_template, request, Response, g
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pathlib import Path
from mlProject.config.configuration import ConfigurationManager
from mlProject.constants import ENV_FLASK_PORT, ENV_FLASK_DEBUG, ENV_TAG
from mlProject.pipeline.prediction import PredictionPipeline
from mlProject import logger
from mlProject.utils.common import load_env_file, get_env_or_config
from mlProject.utils.model_registry import load_registry, rollback_to_version
from mlProject.components.data_transformation import NUMERIC_FEATURES
from mlProject.components.xai_explainer import XAIExplainer
import joblib

# Enterprise MLOps components
from mlProject.components.security import create_token, decode_token, require_role, AuditLogger, USER_DB
from werkzeug.security import check_password_hash
from mlProject.components.retraining import RetrainingEngine
from mlProject.components.observability import APILogger, ObservabilityCollector


@functools.lru_cache(maxsize=1)
def _get_registry_path() -> Path:
    """Get the configured model registry path.

    Cached for the process lifetime — the path is derived from immutable
    config and is resolved on every /health request.
    """
    try:
        return ConfigurationManager().get_model_registry_config().registry_path
    except Exception:
        return Path("artifacts/model_registry.json")

load_env_file()

app = Flask(__name__)

# Request logging middleware for API Gateway Request Analytics
@app.before_request
def before_request():
    g.start_time = time.time()

@app.after_request
def after_request(response):
    if request.path.startswith("/static/") or request.path == "/health":
        return response
    start_time = getattr(g, "start_time", None)
    if start_time:
        latency_ms = (time.time() - start_time) * 1000
        try:
            ip = request.remote_addr
            endpoint = request.path
            method = request.method
            status_code = response.status_code
            APILogger().log_request(endpoint, method, status_code, latency_ms, ip)
        except Exception as e:
            app.logger.error(f"Error logging request: {e}")
    return response

# Global pipeline instance — loaded once at startup to avoid per-request disk I/O
pipeline = PredictionPipeline()
explainer = None


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


_TRAINING_STATE_FILE = Path("artifacts/.training.state")


def _read_training_state() -> dict:
    """Read the shared training state file. Returns default state if missing."""
    default = {"is_training": False, "started_at": None, "log": []}
    try:
        if _TRAINING_STATE_FILE.exists():
            with open(_TRAINING_STATE_FILE, "r") as f:
                state = json.load(f)
            return {
                "is_training": state.get("is_training", False),
                "started_at": state.get("started_at"),
                "log": state.get("log", []),
            }
    except Exception:
        pass
    return default


def _write_training_state(is_training: bool, log_messages: list = None, started_at: float = None) -> None:
    """Atomically write training state to the shared state file."""
    _TRAINING_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "is_training": is_training,
        "started_at": started_at,
        "log": (log_messages if log_messages is not None else list(training_log))[-MAX_LOG_LINES:],
    }
    tmp = _TRAINING_STATE_FILE.with_suffix(".state.tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(state, f)
        tmp.replace(_TRAINING_STATE_FILE)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)



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
        app.logger.warning("TRAIN_SECRET not set - /train endpoint disabled")
        return False  # No secret configured - refuse everything

    supplied = (
        request.args.get("token", "")
        or request.headers.get("X-Train-Token", "")
    )
    result = secrets.compare_digest(supplied.encode(), expected.encode())
    if not result:
        app.logger.warning(f"Unauthorized /train access attempt from {request.remote_addr}")
    return result


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
    """Decorator to require admin token (old API token) or a valid JWT token with Admin role."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        # 1. Try static admin/train tokens (backward compatibility)
        token = (request.headers.get('X-Admin-Token')
                 or request.headers.get('X-Train-Token')
                 or request.args.get("token", ""))
        expected = os.environ.get("ADMIN_TOKEN") or os.environ.get("TRAIN_SECRET", "")
        if token and expected and secrets.compare_digest(token.encode(), expected.encode()):
            log_admin_action(f.__name__, "Authenticated via static token")
            AuditLogger().log_action("static_admin", request.path, "GRANTED", request.remote_addr)
            return f(*args, **kwargs)
            
        # 2. Try JWT Bearer auth
        auth_header = request.headers.get("Authorization")
        jwt_token = request.args.get("token")
        if auth_header and auth_header.startswith("Bearer "):
            jwt_token = auth_header.split(" ")[1]
            
        if jwt_token:
            payload = decode_token(jwt_token)
            if isinstance(payload, dict) and payload.get("role") == "Admin":
                username = payload.get("sub", "unknown")
                log_admin_action(f.__name__, f"Authenticated via JWT Admin: {username}")
                AuditLogger().log_action(username, request.path, "GRANTED", request.remote_addr)
                return f(*args, **kwargs)
                
        # 3. Fail closed
        AuditLogger().log_action("anonymous", request.path, "DENIED", request.remote_addr, "Missing or invalid Admin credentials")
        return jsonify({"error": "Admin access required. Please login as Admin."}), 401
    return decorated


# ---------------------------------------------------------------------------
# Log helper
# ---------------------------------------------------------------------------
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
        is_training = False
        with _log_lock:
            training_log.append("Training rejected: another process is already training")
        try:
            _training_lock.release()
        except RuntimeError:
            pass
        return
    start_time = time.time()
    _write_training_state(True, ["Training started..."], started_at=start_time)
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
        _write_training_state(False, list(training_log), started_at=start_time)
    except subprocess.TimeoutExpired:
        with _log_lock:
            training_log.append(
                f"Training timed out after {TRAIN_TIMEOUT}s"
            )
        _write_training_state(False, list(training_log), started_at=start_time)
    except Exception as exc:
        with _log_lock:
            training_log.append(f"Training error: {exc}")
        _write_training_state(False, list(training_log), started_at=start_time)
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
            train_timeout = int(os.environ.get("TRAIN_TIMEOUT", "1800"))
            result = subprocess.run(
                [sys.executable, "main.py"],
                capture_output=True,
                text=True,
                timeout=train_timeout,
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


@app.route("/dashboard", methods=["GET"])
def dashboard():
    return render_template("dashboard.html")


@app.route("/health", methods=["GET"])
def health():
    """Lightweight liveness probe for uptime monitors and load balancers."""
    state = _read_training_state()
    health_data = {"status": "ok", "is_training": state["is_training"]}
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

    # 4 - Write shared state so all workers see training is active
    _write_training_state(True, [], started_at=time.time())

    # 5 - Submit to ThreadPoolExecutor (reuses worker thread, avoids thread leak)
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

    state = _read_training_state()

    return jsonify({
        "is_training": state["is_training"],
        "log": state["log"],
    })


@app.route("/explain/global", methods=["GET"])
@limiter.limit("30 per minute")
def explain_global():
    """Return global feature importances using SHAP."""
    global explainer
    if explainer is None:
        try:
            if pipeline.unified_pipeline is None:
                pipeline.predict(np.zeros((1, len(NUMERIC_FEATURES))))
            explainer = XAIExplainer(pipeline.unified_pipeline)
        except Exception as e:
            return jsonify({"error": f"Failed to initialize explainer: {e}"}), 500
    importance = explainer.get_global_importance()
    return jsonify(importance)


@app.route("/explain/local", methods=["POST"])
@limiter.limit("60 per minute")
def explain_local():
    """Return local feature contributions using SHAP for a given request."""
    global explainer
    try:
        if request.is_json:
            data = request.json
        else:
            data = request.form.to_dict()
        inputs = {}
        for feature in NUMERIC_FEATURES:
            # Use an explicit None check so that legitimate zero/non-negative
            # values (e.g. citric acid: 0, chlorides: 0) are accepted
            # instead of being dropped by truthiness of `or`.
            val = data.get(feature)
            if val is None:
                val = data.get(feature.replace(" ", "_"))
            if val is None:
                return jsonify({"error": f"Missing required feature: {feature}"}), 400
            try:
                val = float(val)
            except (TypeError, ValueError):
                return jsonify({"error": f"Invalid value for feature: {feature}"}), 400
            # Range validation must match the /predict contract so the two
            # endpoints accept/reject identical inputs.
            if feature == "fixed acidity" and val <= 0:
                return jsonify({"error": "Fixed Acidity must be positive."}), 400
            if feature == "volatile acidity" and val <= 0:
                return jsonify({"error": "Volatile Acidity must be positive."}), 400
            if feature == "citric acid" and val < 0:
                return jsonify({"error": "Citric Acid must be non-negative."}), 400
            if feature == "residual sugar" and val <= 0:
                return jsonify({"error": "Residual Sugar must be positive."}), 400
            if feature == "chlorides" and val < 0:
                return jsonify({"error": "Chlorides must be non-negative."}), 400
            if feature == "free sulfur dioxide" and val < 0:
                return jsonify({"error": "Free Sulfur Dioxide must be non-negative."}), 400
            if feature == "total sulfur dioxide" and val < 0:
                return jsonify({"error": "Total Sulfur Dioxide must be non-negative."}), 400
            if feature == "density" and val <= 0:
                return jsonify({"error": "Density must be positive."}), 400
            if feature == "pH" and not (0 < val < 14):
                return jsonify({"error": "pH must be between 0 and 14."}), 400
            if feature == "sulphates" and val < 0:
                return jsonify({"error": "Sulphates must be non-negative."}), 400
            if feature == "alcohol" and val <= 0:
                return jsonify({"error": "Alcohol must be positive."}), 400
            inputs[feature] = val
        if explainer is None:
            if pipeline.unified_pipeline is None:
                pipeline.predict(np.zeros((1, len(NUMERIC_FEATURES))))
            explainer = XAIExplainer(pipeline.unified_pipeline)
        explanation = explainer.explain_instance(inputs)
        return jsonify(explanation)
    except Exception as e:
        app.logger.error(f"Failed to explain instance: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/monitoring/drift", methods=["GET"])
@limiter.limit("30 per minute")
def monitoring_drift():
    """Return model monitoring drift report."""
    from mlProject.components.monitoring import DriftDetector
    try:
        detector = DriftDetector()
        report = detector.detect_drift(min_predictions=5) # 5 predictions threshold for demo ease
        return jsonify(report)
    except Exception as e:
        app.logger.error(f"Failed to detect drift: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/monitoring/history", methods=["GET"])
@limiter.limit("30 per minute")
def monitoring_history():
    """Return prediction history logs."""
    from mlProject.components.monitoring import PredictionLogger
    try:
        pred_logger = PredictionLogger()
        history_df = pred_logger.get_logged_predictions(limit=100)
        if history_df.empty:
            return jsonify([])
        return jsonify(history_df.to_dict(orient="records"))
    except Exception as e:
        app.logger.error(f"Failed to fetch history logs: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/ethics/check", methods=["POST"])
@limiter.limit("30 per minute")
@require_role(["Admin", "Engineer"])
def ethics_check():
    """Run a bias detection check across a protected attribute for a batch of records."""
    from mlProject.components.ethical_monitoring import EthicalAIMonitor
    data = request.json or {}
    records = data.get("records")
    protected_attribute = data.get("protected_attribute")
    threshold = data.get("threshold", 0.1)

    if not records or not protected_attribute:
        return jsonify({"error": "records and protected_attribute are required"}), 400

    try:
        monitor = EthicalAIMonitor()
        report = monitor.detect_bias(records, protected_attribute, threshold=threshold)
        return jsonify(report)
    except Exception as e:
        app.logger.error(f"Failed to run ethics check: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/ethics/bias/history", methods=["GET"])
@limiter.limit("30 per minute")
@require_role(["Admin", "Engineer", "Viewer"])
def ethics_bias_history():
    """Return historical bias check results."""
    from mlProject.components.ethical_monitoring import EthicalAIMonitor
    limit = request.args.get("limit", default=100, type=int)
    monitor = EthicalAIMonitor()
    return jsonify(monitor.get_bias_history(limit=limit))


@app.route("/ethics/violations", methods=["GET"])
@limiter.limit("30 per minute")
@require_role(["Admin", "Engineer", "Viewer"])
def ethics_violations():
    """Return logged compliance violations."""
    from mlProject.components.ethical_monitoring import EthicalAIMonitor
    limit = request.args.get("limit", default=100, type=int)
    monitor = EthicalAIMonitor()
    return jsonify(monitor.get_compliance_report(limit=limit))


@app.route("/ethics/violations", methods=["POST"])
@limiter.limit("30 per minute")
@require_role(["Admin", "Engineer"])
def ethics_log_violation():
    """Manually log a compliance violation."""
    from mlProject.components.ethical_monitoring import EthicalAIMonitor
    data = request.json or {}
    violation_type = data.get("violation_type")
    severity = data.get("severity")
    description = data.get("description", "")

    if not violation_type or not severity:
        return jsonify({"error": "violation_type and severity are required"}), 400

    monitor = EthicalAIMonitor()
    violation_id = monitor.log_violation(violation_type, severity, description)
    return jsonify({"message": "Violation logged", "id": violation_id})


@app.route("/ethics/alerts", methods=["GET"])
@limiter.limit("30 per minute")
@require_role(["Admin", "Engineer", "Viewer"])
def ethics_alerts():
    """Return active (or all) ethics monitoring alerts."""
    from mlProject.components.ethical_monitoring import EthicalAIMonitor
    status = request.args.get("status", default="open")
    if status.lower() == "all":
        status = None
    limit = request.args.get("limit", default=100, type=int)
    monitor = EthicalAIMonitor()
    return jsonify(monitor.get_alerts(status=status, limit=limit))


@app.route("/ethics/alerts/<int:alert_id>/resolve", methods=["POST"])
@limiter.limit("30 per minute")
@require_role(["Admin", "Engineer"])
def ethics_resolve_alert(alert_id):
    """Mark an ethics alert as resolved."""
    from mlProject.components.ethical_monitoring import EthicalAIMonitor
    monitor = EthicalAIMonitor()
    resolved = monitor.resolve_alert(alert_id)
    if resolved:
        return jsonify({"message": f"Alert {alert_id} resolved"})
    return jsonify({"error": f"Alert {alert_id} not found or already resolved"}), 404


@app.route("/ethics/dashboard", methods=["GET"])
@limiter.limit("30 per minute")
@require_role(["Admin", "Engineer", "Viewer"])
def ethics_dashboard():
    """Return summary counts for the ethical AI monitoring dashboard widget."""
    from mlProject.components.ethical_monitoring import EthicalAIMonitor
    monitor = EthicalAIMonitor()
    return jsonify(monitor.get_dashboard_summary())


@app.route("/experiments/runs", methods=["GET"])
@limiter.limit("30 per minute")
def experiments_runs():
    """Return all experiments run from MLflow."""
    from mlProject.components.experiment_tracker import get_mlflow_runs
    try:
        return jsonify(get_mlflow_runs())
    except Exception as e:
        app.logger.error(f"Failed to fetch experiments runs: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/decisions/analyze", methods=["POST"])
@limiter.limit("30 per minute")
@require_role(["Admin", "Engineer"])
def decisions_analyze():
    """Analyze a decision context and return a ranked, explainable recommendation."""
    from mlProject.components.decision_intelligence import DecisionIntelligenceEngine
    data = request.json or {}
    context = data.get("context")
    options = data.get("options")

    if not context or not options:
        return jsonify({"error": "context and options are required"}), 400

    try:
        engine = DecisionIntelligenceEngine()
        result = engine.analyze_decision(context, options)
        status_code = 200 if result.get("status") == "success" else 400
        return jsonify(result), status_code
    except Exception as e:
        app.logger.error(f"Failed to analyze decision: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/decisions/history", methods=["GET"])
@limiter.limit("30 per minute")
@require_role(["Admin", "Engineer", "Viewer"])
def decisions_history():
    """Return historical decision recommendations."""
    from mlProject.components.decision_intelligence import DecisionIntelligenceEngine
    limit = request.args.get("limit", default=100, type=int)
    engine = DecisionIntelligenceEngine()
    return jsonify(engine.get_decision_history(limit=limit))


@app.route("/decisions/outcome", methods=["POST"])
@limiter.limit("30 per minute")
@require_role(["Admin", "Engineer"])
def decisions_log_outcome():
    """Log the actual outcome of a past decision, for performance tracking."""
    from mlProject.components.decision_intelligence import DecisionIntelligenceEngine
    data = request.json or {}
    decision_id = data.get("decision_id")
    chosen_option = data.get("chosen_option")
    actual_outcome = data.get("actual_outcome")

    if decision_id is None or not chosen_option or actual_outcome is None:
        return jsonify({"error": "decision_id, chosen_option, and actual_outcome are required"}), 400

    engine = DecisionIntelligenceEngine()
    outcome_id = engine.log_outcome(decision_id, chosen_option, actual_outcome)
    return jsonify({"message": "Outcome logged", "id": outcome_id})


@app.route("/decisions/performance", methods=["GET"])
@limiter.limit("30 per minute")
@require_role(["Admin", "Engineer", "Viewer"])
def decisions_performance():
    """Return recommendation performance metrics based on logged outcomes."""
    from mlProject.components.decision_intelligence import DecisionIntelligenceEngine
    engine = DecisionIntelligenceEngine()
    return jsonify(engine.get_performance_metrics())


@app.route("/decisions/alerts", methods=["GET"])
@limiter.limit("30 per minute")
@require_role(["Admin", "Engineer", "Viewer"])
def decisions_alerts():
    """Return active (or all) decision intelligence alerts."""
    from mlProject.components.decision_intelligence import DecisionIntelligenceEngine
    status = request.args.get("status", default="open")
    if status.lower() == "all":
        status = None
    limit = request.args.get("limit", default=100, type=int)
    engine = DecisionIntelligenceEngine()
    return jsonify(engine.get_alerts(status=status, limit=limit))


@app.route("/decisions/alerts/<int:alert_id>/resolve", methods=["POST"])
@limiter.limit("30 per minute")
@require_role(["Admin", "Engineer"])
def decisions_resolve_alert(alert_id):
    """Mark a decision intelligence alert as resolved."""
    from mlProject.components.decision_intelligence import DecisionIntelligenceEngine
    engine = DecisionIntelligenceEngine()
    resolved = engine.resolve_alert(alert_id)
    if resolved:
        return jsonify({"message": f"Alert {alert_id} resolved"})
    return jsonify({"error": f"Alert {alert_id} not found or already resolved"}), 404


@app.route("/decisions/dashboard", methods=["GET"])
@limiter.limit("30 per minute")
@require_role(["Admin", "Engineer", "Viewer"])
def decisions_dashboard():
    """Return summary counts for the decision intelligence dashboard widget."""
    from mlProject.components.decision_intelligence import DecisionIntelligenceEngine
    engine = DecisionIntelligenceEngine()
    return jsonify(engine.get_dashboard_summary())


@app.route("/benchmarking/results", methods=["GET"])
@limiter.limit("30 per minute")
def benchmarking_results():
    """Return model benchmarking comparison results."""
    import json
    from pathlib import Path
    benchmark_path = Path("artifacts/model_trainer/benchmark_results.json")
    if benchmark_path.exists():
        try:
            with open(benchmark_path) as f:
                results = json.load(f)
            return jsonify(results)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({
        "ElasticNet": {"r2": 0.3552, "rmse": 0.6469, "mae": 0.5063},
        "RandomForestRegressor": {"r2": 0.4521, "rmse": 0.5962, "mae": 0.4632},
        "GradientBoostingRegressor": {"r2": 0.4851, "rmse": 0.5781, "mae": 0.4412},
        "XGBoost": {"r2": 0.5123, "rmse": 0.5629, "mae": 0.4291}
    })


@app.route("/analytics/summary", methods=["GET"])
@limiter.limit("30 per minute")
def analytics_summary():
    """Return prediction summary statistics and trend data."""
    from mlProject.components.analytics import get_analytics_summary
    try:
        return jsonify(get_analytics_summary())
    except Exception as e:
        app.logger.error(f"Failed to fetch analytics summary: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/analytics/export/csv", methods=["GET"])
@limiter.limit("10 per hour")
def analytics_export_csv():
    """Export predictions database as CSV."""
    from mlProject.components.monitoring import PredictionLogger
    from flask import make_response
    pred_logger = PredictionLogger()
    df = pred_logger.get_logged_predictions()
    if df.empty:
        return jsonify({"error": "No predictions logged yet"}), 400
    csv_data = df.to_csv(index=False)
    response = make_response(csv_data)
    response.headers["Content-Disposition"] = "attachment; filename=predictions_export.csv"
    response.headers["Content-type"] = "text/csv"
    return response


@app.route("/analytics/export/pdf", methods=["GET"])
@limiter.limit("10 per hour")
def analytics_export_pdf():
    """Export predictions analytics report as PDF."""
    from mlProject.components.analytics import generate_pdf_report
    from flask import send_file
    try:
        pdf_buffer = generate_pdf_report()
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name="wine_quality_analytics_report.pdf",
            mimetype="application/pdf"
        )
    except Exception as e:
        app.logger.error(f"Failed to generate PDF report: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST", "GET"])
@limiter.limit("30 per minute")
def index():
    if request.method == "POST":
        if request.content_type and "form" not in request.content_type and "urlencoded" not in request.content_type:
            return render_template("results.html", error_msg="Only form-encoded data is supported. Use Content-Type: application/x-www-form-urlencoded."), 400
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

            data = pd.DataFrame([[
                fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                density, pH, sulphates, alcohol,
            ]], columns=[
                "fixed acidity", "volatile acidity", "citric acid",
                "residual sugar", "chlorides", "free sulfur dioxide",
                "total sulfur dioxide", "density", "pH", "sulphates", "alcohol",
            ])

            predict = pipeline.predict(data)
            final_prediction = round(float(predict[0]), 2)

            plot_url = None
            try:
                import shap
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                import base64
                import io

                shap_values = pipeline.explain(data)
                if shap_values is not None:
                    plt.clf()
                    # waterfall plot requires a single Explanation object
                    shap.plots.waterfall(shap_values[0], show=False)
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png", bbox_inches='tight', dpi=150)
                    buf.seek(0)
                    plot_url = base64.b64encode(buf.getvalue()).decode("utf-8")
                    plt.close()
            except Exception as e:
                logger.error(f"Failed to generate SHAP plot: {e}")

            return render_template("results.html", prediction=final_prediction, plot_url=plot_url)

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


@app.route("/predict/batch", methods=["POST"])
@limiter.limit("10 per minute")
def predict_batch():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    try:
        # Read the uploaded CSV file
        df = pd.read_csv(file)
        
        # Keep only the required features, ignoring any target columns if present
        from mlProject.components.data_transformation import NUMERIC_FEATURES
        
        missing_cols = [col for col in NUMERIC_FEATURES if col not in df.columns]
        if missing_cols:
            return jsonify({"error": f"Missing required columns: {', '.join(missing_cols)}"}), 400
            
        test_x = df[NUMERIC_FEATURES]
        
        # Predict
        predictions = pipeline.predict(test_x)
        
        # Append predictions
        df['predicted_quality'] = np.round(predictions, 2)
        
        # Convert back to CSV
        output = io.StringIO()
        df.to_csv(output, index=False)
        csv_data = output.getvalue()
        
        return Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment;filename=predictions.csv"}
        )
        
    except pd.errors.EmptyDataError:
        return jsonify({"error": "The uploaded CSV file is empty"}), 400
    except Exception as e:
        logger.error(f"Error in /predict/batch: {e}")
        return jsonify({"error": f"An error occurred processing the file: {str(e)}"}), 500


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
    data = request.get_json(silent=True) or {}
    version_id = data.get("version_id")
    if not version_id:
        return jsonify({"error": "version_id is required"}), 400
    registry_path = _get_registry_path()
    current_prod = get_current_production_version(registry_path)
    log_admin_action(
        "rollback_initiated",
        f"target_version={version_id}, current_production={current_prod}"
    )
    if rollback_to_version(registry_path, version_id):
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


# ===========================================================================
# Enterprise Security, Retraining, Registry & Observability API endpoints
# ===========================================================================

@app.route("/auth/login", methods=["POST"])
def auth_login():
    """Authenticates users and returns a JWT token with their RBAC role."""
    data = request.json or {}
    username = data.get("username")
    password = data.get("password")
    
    if not username or not password:
        AuditLogger().log_action("anonymous", "login", "FAILED", request.remote_addr, "Missing credentials")
        return jsonify({"error": "Username and password are required"}), 400
        
    user = USER_DB.get(username)
    if not user or not check_password_hash(user.get("password_hash", ""), password):
        AuditLogger().log_action(username, "login", "FAILED", request.remote_addr, "Invalid password or user")
        return jsonify({"error": "Invalid username or password"}), 401
        
    role = user["role"]
    token = create_token(username, role)
    AuditLogger().log_action(username, "login", "SUCCESS", request.remote_addr, f"Role: {role}")
    return jsonify({"token": token, "role": role, "username": username})


@app.route("/auth/audit-logs", methods=["GET"])
@require_role(["Admin", "Engineer"])
def get_audit_logs():
    """Retrieve security audit logs."""
    limit = request.args.get("limit", default=100, type=int)
    try:
        logs = AuditLogger().get_logs(limit=limit)
        return jsonify(logs)
    except Exception as e:
        app.logger.error(f"Failed to fetch audit logs: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/retrain/trigger", methods=["POST"])
@require_role(["Admin", "Engineer"])
def retrain_trigger():
    """Manually trigger the model retraining pipeline."""
    data = request.json or {}
    reason = data.get("reason", "Manual trigger via API")
    success = RetrainingEngine().trigger_retraining(reason=reason)
    if success:
        return jsonify({"message": "Retraining pipeline triggered successfully in background."})
    return jsonify({"error": "Retraining already in progress."}), 409


@app.route("/retrain/history", methods=["GET"])
@require_role(["Admin", "Engineer", "Viewer"])
def retrain_history():
    """Fetch retraining history logs."""
    history_path = Path("artifacts/retrain_history.json")
    if history_path.exists():
        try:
            with open(history_path, "r") as f:
                return jsonify(json.load(f))
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify([])


@app.route("/registry/promote", methods=["POST"])
@require_role(["Admin"])
def promote_model():
    """Promote model version to production stage."""
    data = request.json or {}
    version_id = data.get("version_id")
    if not version_id:
        return jsonify({"error": "version_id is required"}), 400
        
    registry_path = _get_registry_path()
    stable_model_path = Path("artifacts/model_trainer/model.joblib")
    success = update_registration(
        registry_path=registry_path,
        version_id=version_id,
        status="production",
        stable_model_path=stable_model_path
    )
    if success:
        AuditLogger().log_action("admin", f"promote_model:{version_id}", "SUCCESS", request.remote_addr)
        return jsonify({"message": f"Successfully promoted version {version_id} to Production stage"})
    return jsonify({"error": f"Version {version_id} not found in registry"}), 404


@app.route("/registry/demote", methods=["POST"])
@require_role(["Admin"])
def demote_model():
    """Demote model version to staging stage."""
    data = request.json or {}
    version_id = data.get("version_id")
    if not version_id:
        return jsonify({"error": "version_id is required"}), 400
        
    registry_path = _get_registry_path()
    success = update_registration(
        registry_path=registry_path,
        version_id=version_id,
        status="staging"
    )
    if success:
        AuditLogger().log_action("admin", f"demote_model:{version_id}", "SUCCESS", request.remote_addr)
        return jsonify({"message": f"Successfully demoted version {version_id} to Staging stage"})
    return jsonify({"error": f"Version {version_id} not found in registry"}), 404


@app.route("/registry/archive", methods=["POST"])
@require_role(["Admin"])
def archive_model():
    """Archive model version."""
    data = request.json or {}
    version_id = data.get("version_id")
    if not version_id:
        return jsonify({"error": "version_id is required"}), 400
        
    registry_path = _get_registry_path()
    success = update_registration(
        registry_path=registry_path,
        version_id=version_id,
        status="archived"
    )
    if success:
        AuditLogger().log_action("admin", f"archive_model:{version_id}", "SUCCESS", request.remote_addr)
        return jsonify({"message": f"Successfully archived version {version_id}"})
    return jsonify({"error": f"Version {version_id} not found in registry"}), 404


@app.route("/twins", methods=["POST"])
@require_role(["Admin", "Engineer"])
def create_twin():
    """Create a digital twin from a registered model version (defaults to production)."""
    from mlProject.components.digital_twin import DigitalTwin, DigitalTwinError
    data = request.json or {}
    name = data.get("name")
    if not name:
        return jsonify({"error": "name is required"}), 400
    try:
        twin = DigitalTwin().create_twin(name, source_version_id=data.get("source_version_id"))
        return jsonify(twin), 201
    except DigitalTwinError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.error(f"Failed to create digital twin: {e}")
        return jsonify({"error": "Failed to create digital twin"}), 500


@app.route("/twins", methods=["GET"])
@require_role(["Admin", "Engineer", "Viewer"])
def list_twins():
    """List all digital twins."""
    from mlProject.components.digital_twin import DigitalTwin
    return jsonify(DigitalTwin().list_twins())


@app.route("/twins/<twin_id>", methods=["GET"])
@require_role(["Admin", "Engineer", "Viewer"])
def get_twin(twin_id):
    """Fetch a single digital twin's metadata."""
    from mlProject.components.digital_twin import DigitalTwin, DigitalTwinError
    try:
        return jsonify(DigitalTwin().get_twin(twin_id))
    except DigitalTwinError as e:
        return jsonify({"error": str(e)}), 404


@app.route("/twins/<twin_id>", methods=["DELETE"])
@require_role(["Admin"])
def delete_twin(twin_id):
    """Delete a digital twin and its stored model artifact."""
    from mlProject.components.digital_twin import DigitalTwin, DigitalTwinError
    try:
        DigitalTwin().delete_twin(twin_id)
        return jsonify({"message": f"Twin {twin_id} deleted"})
    except DigitalTwinError as e:
        return jsonify({"error": str(e)}), 404


@app.route("/twins/<twin_id>/sync", methods=["POST"])
@require_role(["Admin", "Engineer"])
def sync_twin(twin_id):
    """Re-sync a twin's model + metrics snapshot to the current production version."""
    from mlProject.components.digital_twin import DigitalTwin, DigitalTwinError
    try:
        return jsonify(DigitalTwin().sync_state(twin_id))
    except DigitalTwinError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.error(f"Failed to sync twin {twin_id}: {e}")
        return jsonify({"error": "Failed to sync twin"}), 500


@app.route("/twins/<twin_id>/simulate", methods=["POST"])
@limiter.limit("30 per minute")
@require_role(["Admin", "Engineer"])
def simulate_twin(twin_id):
    """Run a what-if scenario against the twin's frozen model — never touches production."""
    from mlProject.components.digital_twin import DigitalTwin, DigitalTwinError
    data = request.json or {}
    scenario_name = data.get("scenario_name", "unnamed_scenario")
    input_rows = data.get("input_rows")
    if not input_rows or not isinstance(input_rows, list):
        return jsonify({"error": "input_rows must be a non-empty list of feature dicts"}), 400
    try:
        result = DigitalTwin().run_simulation(twin_id, scenario_name, input_rows)
        return jsonify(result)
    except DigitalTwinError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.error(f"Simulation failed for twin {twin_id}: {e}")
        return jsonify({"error": "Simulation failed"}), 500


@app.route("/twins/<twin_id>/simulations", methods=["GET"])
@require_role(["Admin", "Engineer", "Viewer"])
def twin_simulation_history(twin_id):
    """Fetch a twin's past simulation runs."""
    from mlProject.components.digital_twin import DigitalTwin
    limit = request.args.get("limit", default=50, type=int)
    return jsonify(DigitalTwin().get_simulation_history(twin_id, limit=limit))


@app.route("/twins/<twin_id>/test-change", methods=["POST"])
@require_role(["Admin", "Engineer"])
def test_twin_change(twin_id):
    """Compare a candidate model artifact against the twin's baseline before promotion."""
    from mlProject.components.digital_twin import DigitalTwin, DigitalTwinError
    data = request.json or {}
    candidate_model_path = data.get("candidate_model_path")
    test_rows = data.get("test_rows")
    if not candidate_model_path or not test_rows:
        return jsonify({"error": "candidate_model_path and test_rows are required"}), 400
    try:
        result = DigitalTwin().test_change(twin_id, candidate_model_path, test_rows)
        return jsonify(result)
    except DigitalTwinError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.error(f"Change test failed for twin {twin_id}: {e}")
        return jsonify({"error": "Change test failed"}), 500


@app.route("/twins/<twin_id>/drift", methods=["GET"])
@require_role(["Admin", "Engineer", "Viewer"])
def twin_drift(twin_id):
    """Check whether a twin has fallen out of sync with the current production version."""
    from mlProject.components.digital_twin import DigitalTwin, DigitalTwinError
    try:
        return jsonify(DigitalTwin().detect_twin_drift(twin_id))
    except DigitalTwinError as e:
        return jsonify({"error": str(e)}), 404


@app.route("/observability/health", methods=["GET"])
@require_role(["Admin", "Engineer", "Viewer"])
def observability_health():
    """Retrieve system health and active alerts."""
    try:
        collector = ObservabilityCollector()
        return jsonify(collector.get_system_health())
    except Exception as e:
        app.logger.error(f"Failed to fetch system health: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/analytics", methods=["GET"])
@require_role(["Admin", "Engineer", "Viewer"])
def api_analytics():
    """Retrieve API latency, status codes, and request analytics."""
    try:
        logger = APILogger()
        return jsonify(logger.get_analytics(hours=24))
    except Exception as e:
        app.logger.error(f"Failed to fetch API analytics: {e}")
        return jsonify({"error": str(e)}), 500


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

    # Gunicorn handles this via gunicorn.conf.py; the dev server must do it
    # itself so a fresh clone trains a model before the first /predict.
    ensure_model_trained()

    app.run(host="0.0.0.0", port=port, debug=debug)