import os
import json
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from datetime import datetime
from mlProject import logger

class RetrainingEngine:
    def __init__(self, db_path="artifacts/predictions.db", history_path="artifacts/retrain_history.json"):
        self.db_path = db_path
        self.history_path = history_path
        self.retraining_in_progress = False
        self._lock = threading.Lock()
        
    def log_retraining_run(self, status, message, metrics=None):
        history = []
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, "r") as f:
                    history = json.load(f)
            except Exception:
                pass
        
        run_info = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": status,
            "message": message,
            "metrics": metrics or {}
        }
        history.insert(0, run_info)
        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
        with open(self.history_path, "w") as f:
            json.dump(history, f, indent=2)

    def trigger_retraining(self, reason="Manual trigger") -> bool:
        with self._lock:
            if self.retraining_in_progress:
                logger.warning("Retraining already in progress. Rejecting trigger.")
                return False
            self.retraining_in_progress = True
            
        def run_thread():
            try:
                logger.info(f"Initiating automated retraining. Reason: {reason}")
                self.log_retraining_run("started", f"Retraining triggered due to: {reason}")
                
                result = subprocess.run(
                    [sys.executable, "main.py"],
                    capture_output=True,
                    text=True,
                    timeout=600
                )
                if result.returncode == 0:
                    logger.info("Automated retraining finished successfully!")
                    metrics_path = Path("artifacts/model_evaluation/metrics.json")
                    eval_metrics = {}
                    if metrics_path.exists():
                        try:
                            with open(metrics_path, "r") as f:
                                eval_metrics = json.load(f)
                        except Exception:
                            pass
                            
                    self.log_retraining_run("success", "Pipeline retraining completed successfully", eval_metrics)
                else:
                    logger.error(f"Automated retraining failed: {result.stderr}")
                    self.log_retraining_run("failed", f"Pipeline failed: {result.stderr or result.stdout}")
            except Exception as e:
                logger.exception(f"Exception during automated retraining: {e}")
                self.log_retraining_run("failed", f"Internal error during execution: {str(e)}")
            finally:
                self.retraining_in_progress = False
                
        t = threading.Thread(target=run_thread)
        t.start()
        return True

    def check_and_trigger_on_drift(self) -> bool:
        """
        Check if drift is detected. If drift is detected on >=20% of features, trigger retraining.
        """
        from mlProject.components.monitoring import DriftDetector
        try:
            detector = DriftDetector(db_path=self.db_path)
            report = detector.detect_drift(min_predictions=5)
            if report.get("status") == "success" and report.get("drift_detected", False):
                ratio = report.get("drifted_features_ratio", 0)
                if ratio >= 0.20:
                    logger.warning(f"Data drift ratio {ratio:.2f} >= threshold 0.20. Triggering automated retraining...")
                    return self.trigger_retraining(reason=f"Data drift detected on {ratio*100:.1f}% features")
        except Exception as e:
            logger.error(f"Error checking drift for retraining trigger: {e}")
        return False
