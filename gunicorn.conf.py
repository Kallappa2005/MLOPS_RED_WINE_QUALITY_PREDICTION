"""Gunicorn configuration with safe single-worker model training."""

import os
import sys
import time
import logging

logger = logging.getLogger(__name__)

bind = os.environ.get("GUNICORN_BIND", "0.0.0.0:8080")
workers = int(os.environ.get("GUNICORN_WORKERS", "2"))
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "120"))
loglevel = os.environ.get("GUNICORN_LOGLEVEL", "info")


def when_ready(server):
    """Called when the master process is ready — before forking workers.
    Uses a PID lock file to ensure only one master runs training.
    Clears stale cross-worker training state before workers start.
    """
    _clear_stale_training_state()
    lock_path = "/tmp/model_init.lock"
    status_path = "/tmp/model_init_done"

    # Remove stale lock file if no gunicorn master process is running
    if os.path.exists(lock_path):
        stale_pid = None
        try:
            with open(lock_path) as f:
                stale_pid = int(f.read().strip())
            os.kill(stale_pid, 0)  # Process still alive, lock is valid
        except (OSError, ValueError):
            server.log.info(f"Removing stale lock file (PID={stale_pid})")
            os.remove(lock_path)

    if os.path.exists(status_path):
        server.log.info("Model already initialised — skipping training")
        return

    pid = str(os.getpid())
    try:
        with open(lock_path, "x") as f:
            f.write(pid)
    except FileExistsError:
        server.log.info("Another process holds the training lock — skipping")
        return

    try:
        _ensure_model_trained(server)
        with open(status_path, "w") as f:
            f.write("done")
    finally:
        if os.path.exists(lock_path):
            os.remove(lock_path)


def _clear_stale_training_state():
    """Remove any stale training state file left from a previous run."""
    state_file = os.path.join("artifacts", ".training.state")
    try:
        if os.path.exists(state_file):
            os.remove(state_file)
    except Exception:
        pass


def _ensure_model_trained(server):
    """Train the model if no artifact exists, verify integrity otherwise."""
    try:
        from mlProject.config.configuration import ConfigurationManager
        config_manager = ConfigurationManager()
        eval_config = config_manager.get_model_evaluation_config()
        model_path = eval_config.model_path
    except Exception:
        from pathlib import Path
        model_path = Path("artifacts/model_trainer/model.joblib")

    if model_path.exists():
        from mlProject.utils.common import verify_model_integrity
        checksum_path = model_path.with_suffix(model_path.suffix + ".sha256")
        if not verify_model_integrity(model_path, checksum_path):
            server.log.error("Model integrity check FAILED — consider retraining")
        else:
            server.log.info("Model exists and verified — ready for predictions")
        return

    server.log.info("Model not found — starting automatic training (single master)...")
    try:
        import subprocess
        result = subprocess.run(
            ["python", "main.py"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            server.log.info("Auto-training completed successfully")
        else:
            server.log.error(f"Auto-training failed:\n{result.stderr}")
    except Exception as exc:
        server.log.error(f"Auto-training failed: {exc}")
