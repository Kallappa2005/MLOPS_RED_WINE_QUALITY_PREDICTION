import os
import json
import random
from datetime import datetime

class HyperparameterOptimizer:
    def __init__(self, history_path="artifacts/optimization_history.json"):
        self.history_path = history_path

    def run_optimization_sweep(self, model_type: str, search_strategy="random", iterations=5) -> dict:
        """
        Simulate a hyperparameter optimization sweep and log results.
        """
        results = []
        best_r2 = -1.0
        best_params = {}

        # Default elasticnet tuning simulation
        if model_type == "ElasticNet":
            for i in range(iterations):
                alpha = round(random.uniform(0.1, 1.0), 3)
                l1_ratio = round(random.uniform(0.1, 0.9), 3)
                
                # Simulate R2 score improvement correlation
                r2 = round(0.32 + (alpha * 0.1) + (l1_ratio * 0.05) + random.uniform(-0.02, 0.02), 4)
                rmse = round(0.70 - (r2 * 0.1), 4)
                
                trial_res = {"trial": i + 1, "params": {"alpha": alpha, "l1_ratio": l1_ratio}, "metrics": {"r2": r2, "rmse": rmse}}
                results.append(trial_res)
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_params = trial_res["params"]
        else:
            # Default tree parameters tuning simulation
            for i in range(iterations):
                n_estimators = int(random.choice([50, 100, 200]))
                max_depth = int(random.choice([3, 5, 8, 12]))
                
                r2 = round(0.45 + (max_depth * 0.005) + random.uniform(-0.02, 0.02), 4)
                rmse = round(0.60 - (r2 * 0.1), 4)
                
                trial_res = {"trial": i + 1, "params": {"n_estimators": n_estimators, "max_depth": max_depth}, "metrics": {"r2": r2, "rmse": rmse}}
                results.append(trial_res)
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_params = trial_res["params"]

        sweep_report = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_type": model_type,
            "search_strategy": search_strategy,
            "best_params": best_params,
            "best_r2": best_r2,
            "trials": results
        }

        # Save sweep history
        self._log_sweep(sweep_report)
        return sweep_report

    def _log_sweep(self, sweep_report: dict):
        history = []
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, "r") as f:
                    history = json.load(f)
            except Exception:
                pass
        history.insert(0, sweep_report)
        os.makedirs(os.path.dirname(self.history_path), exist_ok=True)
        with open(self.history_path, "w") as f:
            json.dump(history, f, indent=2)
            
    def get_sweep_history(self) -> list:
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, "r") as f:
                    return json.load(f)
            except Exception:
                return []
        return []
