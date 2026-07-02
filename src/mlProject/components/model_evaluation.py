import json
import os
import shutil
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import numpy as np
import joblib
from mlProject import logger
from mlProject.config.configuration import ConfigurationManager
from mlProject.entity.config_entity import ModelEvaluationConfig
from mlProject.utils.common import save_json, save_checksum
from mlProject.utils.model_registry import (
    load_registry, register_model, update_registration,
)
from mlProject.utils.mlflow_tracker import MlflowTracker
from mlProject.components.data_transformation import NUMERIC_FEATURES
from pathlib import Path


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    
    def baseline_r2_score(self, actual):
        """
        Calculate the R² score for a naive baseline model that always predicts the mean.
        R² < 0 indicates the model performs worse than predicting the mean.
        """
        mean_pred = np.full_like(actual, np.mean(actual))
        baseline_r2 = r2_score(actual, mean_pred)
        return baseline_r2

    def save_results(self):
        try:
            test_data = pd.read_csv(self.config.test_data_path)
        except FileNotFoundError:
            logger.error(f"Test data file not found: {self.config.test_data_path}")
            raise
        except Exception as e:
            logger.exception("Failed to load test data")
            raise

        model_info = self._load_model_info()
        version_id = model_info.get("version_id")
        if not version_id:
            raise ValueError(
                "model_info.json is missing or incomplete (no version_id). "
                "Run model training before evaluation."
            )

        versioned_model_path = model_info.get("model_path")
        if not versioned_model_path:
            raise ValueError(
                "model_info.json is missing model_path. "
                "Run model training before evaluation."
            )

        versioned_model_path = Path(versioned_model_path)
        if not versioned_model_path.exists():
            raise FileNotFoundError(
                f"Versioned model file not found at {versioned_model_path}. "
                f"Run model training before evaluation."
            )

        try:
            from mlProject.utils.common import verify_model_integrity
            checksum_path = Path(str(versioned_model_path) + ".sha256")
            if not verify_model_integrity(versioned_model_path, checksum_path):
                raise ValueError(f"Model integrity check failed for {versioned_model_path}")
            model = joblib.load(versioned_model_path)
        except FileNotFoundError:
            logger.error(f"Model file not found: {versioned_model_path}")
            raise
        except Exception as e:
            logger.exception("Failed to load model")
            raise

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        # Select only raw NUMERIC_FEATURES — the model is a Pipeline whose first
        # step is the preprocessor, so it expects the original 11 features.
        test_x = test_x[NUMERIC_FEATURES]

        try:
            predicted_qualities = model.predict(test_x)
        except Exception as e:
            logger.exception("Model prediction failed")
            raise

        (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)
        
        # Calculate baseline R² (predict-mean strategy)
        baseline_r2 = self.baseline_r2_score(test_y.values.flatten())
        
        # Validate model performance against baseline
        if r2 < 0.0:
            logger.error(f"Model R²={r2:.4f} is below baseline (R²=0.0). Aborting deployment.")
            raise ValueError(f"Model R²={r2:.4f} is below baseline. Aborting.")
        
        scores = {
            "rmse": rmse, 
            "mae": mae, 
            "r2": r2,
            "baseline_r2": baseline_r2
        }

        # Compute per-class metrics
        per_class_metrics = self._compute_per_class_metrics(
            test_data[self.config.target_column].values,
            predicted_qualities,
        )
        if per_class_metrics:
            scores["per_class"] = per_class_metrics

        save_json(path=Path(self.config.metric_file_name), data=scores)

        logger.info(f"Evaluation metrics saved: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}, Baseline R2={baseline_r2:.4f}")

        config_manager = ConfigurationManager()
        registry_config = config_manager.get_model_registry_config()
        registry_path = registry_config.registry_path
        quality_gate = registry_config.quality_gate_max_rmse_degradation_pct

        # Capture the previously-deployed production metrics before registering
        # the new version, otherwise the new version becomes production and we
        # would compare it against itself.
        previous_metrics = self._load_previous_metrics(registry_path)

        params = model_info.get("params", {})
        data_hash = model_info.get("data_hash", "")

        updated = update_registration(
            registry_path=registry_path,
            version_id=version_id,
            metrics=scores,
            params=params,
            data_hash=data_hash,
            quality_gate_max_rmse_degradation_pct=quality_gate,
            status="evaluated",
            model_path=versioned_model_path,
            stable_model_path=self.config.model_path,
        )
        if not updated:
            register_model(
                registry_path=registry_path,
                model_path=versioned_model_path,
                version_id=version_id,
                metrics=scores,
                params=params,
                data_hash=data_hash,
                quality_gate_max_rmse_degradation_pct=quality_gate,
                stable_model_path=self.config.model_path,
            )

        # Promote versioned model to stable path only if quality gate passed
        registry = load_registry(registry_path)
        if registry.get("production") == version_id:
            stable_path = self.config.model_path
            stable_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(versioned_model_path), str(stable_path))
            stable_checksum_path = Path(str(stable_path) + ".sha256")
            save_checksum(stable_path, stable_checksum_path)
            logger.info(f"Model {version_id} promoted to stable path: {stable_path}")

        if previous_metrics:
            comparison = self._compare_metrics(scores, previous_metrics)
            comparison_path = self.config.root_dir / "metrics_comparison.json"
            save_json(path=Path(comparison_path), data=comparison)
            logger.info(f"Metrics comparison saved to {comparison_path}")

        self._log_metrics_to_mlflow(scores, version_id)

    def _log_metrics_to_mlflow(self, scores: dict, version_id: str):
        try:
            config_manager = ConfigurationManager()
            registry_config = config_manager.get_model_registry_config()
            if not registry_config.use_mlflow:
                return
            tracker = MlflowTracker(
                tracking_uri=registry_config.mlflow_tracking_uri,
                experiment_name=registry_config.mlflow_experiment_name,
                use_mlflow=True,
                registry_uri=registry_config.mlflow_registry_uri or None,
            )
            if tracker.start_run(run_name=f"evaluate_{version_id}"):
                tracker.log_metrics({
                    "rmse": scores["rmse"],
                    "mae": scores["mae"],
                    "r2": scores["r2"],
                    "baseline_r2": scores.get("baseline_r2", 0),
                })
                if version_id:
                    tracker.register_model_version(
                        model_name=registry_config.mlflow_model_name,
                        source="model",
                    )
                    registry = load_registry(registry_config.registry_path)
                    prod_id = registry.get("production")
                    if prod_id == version_id:
                        tracker.transition_model_stage(
                            model_name=registry_config.mlflow_model_name,
                            version=version_id,
                            stage="Production",
                        )
                    else:
                        for v in registry.get("versions", []):
                            if v.get("id") == version_id:
                                status = v.get("status", "Staging")
                                tracker.transition_model_stage(
                                    model_name=registry_config.mlflow_model_name,
                                    version=version_id,
                                    stage=status.capitalize(),
                                )
                                break
                tracker.end_run()
                logger.info(f"Metrics logged to MLflow for version {version_id}")
        except Exception as e:
            logger.warning(f"Failed to log metrics to MLflow: {e}")

    def _load_model_info(self) -> dict:
        """Load model_info.json saved alongside the model by model_trainer."""
        model_dir = self.config.model_path.parent
        info_path = model_dir / "model_info.json"
        if info_path.exists():
            try:
                with open(info_path) as f:
                    info = json.load(f)
                if not info.get("version_id") or not info.get("model_path"):
                    raise ValueError(
                        f"model_info.json at {info_path} is incomplete. "
                        f"Required fields: version_id, model_path"
                    )
                return info
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse model_info.json: {e}")
            except Exception as e:
                if isinstance(e, ValueError):
                    raise
                raise ValueError(f"Failed to read model_info.json: {e}")
        raise FileNotFoundError(
            f"Model info not found at {info_path}. "
            f"Run model training before evaluation."
        )

    def _load_previous_metrics(self, registry_path: Path):
        """Load metrics from the previous production model."""
        registry = load_registry(registry_path)
        production_id = registry.get("production")
        if not production_id:
            return None
        for v in registry.get("versions", []):
            if v.get("id") == production_id and v.get("metrics"):
                return v["metrics"]
        return None

    def _compare_metrics(self, current: dict, previous: dict) -> dict:
        """Compare current metrics against previous (skips nested dicts like per_class)."""
        comparison = {"current": current, "previous": previous, "changes": {}}
        for key in current:
            if isinstance(current[key], dict):
                continue
            if key in previous and isinstance(previous[key], (int, float)) and previous[key] != 0:
                pct_change = ((current[key] - previous[key]) / abs(previous[key])) * 100
                comparison["changes"][key] = round(pct_change, 2)
        return comparison

    def _compute_per_class_metrics(self, y_true, y_pred, min_samples=3):
        """Compute per-class RMSE, MAE for each quality level."""
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        per_class = {}
        for quality in sorted(set(y_true)):
            mask = y_true == quality
            count = int(mask.sum())
            if count >= min_samples:
                rmse = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
                mae = mean_absolute_error(y_true[mask], y_pred[mask])
                per_class[int(quality)] = {
                    "rmse": round(float(rmse), 4),
                    "mae": round(float(mae), 4),
                    "count": count,
                }
        return per_class
