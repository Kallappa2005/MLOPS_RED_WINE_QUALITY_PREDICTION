import json
import os
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from mlProject import logger
from mlProject.entity.config_entity import ModelTrainerConfig
from mlProject.components.data_transformation import NUMERIC_FEATURES
from mlProject.utils.mlflow_tracker import MlflowTracker
from mlProject.utils.model_registry import get_version_id, compute_file_hash
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.mlflow_tracker = None

    def train(self):
        try:
            train_data = pd.read_csv(self.config.train_data_path)
            test_data = pd.read_csv(self.config.test_data_path)
        except FileNotFoundError as e:
            logger.error(f"Training/testing data file not found: {e.filename}")
            raise
        except Exception as e:
            logger.exception("Failed to load training/testing data")
            raise

        train_x = train_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]].values.ravel()
        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]].values.ravel()

        # Load preprocessor if available
        preprocessor = None
        preprocessor_path = self.config.preprocessor_path or Path('artifacts/data_transformation/preprocessor.joblib')
        if self.config.use_scaler and preprocessor_path.exists():
            try:
                preprocessor = joblib.load(preprocessor_path)
                logger.info(f"Loaded preprocessor from {preprocessor_path}")
            except Exception as e:
                logger.warning(f"Failed to load preprocessor: {e}. Training model without preprocessor.")

        if preprocessor is not None:
            expected_cols = len(NUMERIC_FEATURES)
            if train_x.shape[1] != expected_cols:
                logger.warning(
                    f"train_x has {train_x.shape[1]} columns but preprocessor "
                    f"expects {expected_cols}. Selecting NUMERIC_FEATURES."
                )
                train_x = train_x[NUMERIC_FEATURES]
                test_x = test_x[NUMERIC_FEATURES]

            train_x_preprocessed = preprocessor.transform(train_x)
            test_x_preprocessed = preprocessor.transform(test_x)
        else:
            train_x_preprocessed = train_x
            test_x_preprocessed = test_x

        # Define candidate models
        models = {
            "ElasticNet": ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
        }

        best_model_name = None
        best_model = None
        best_r2 = -float("inf")
        best_pipeline = None
        benchmark_results = {}

        for name, model in models.items():
            logger.info(f"Training and benchmarking model: {name}")
            try:
                model.fit(train_x_preprocessed, train_y)
                preds = model.predict(test_x_preprocessed)
                r2 = r2_score(test_y, preds)
                rmse = np.sqrt(mean_squared_error(test_y, preds))
                mae = np.mean(np.abs(preds - test_y))
                
                logger.info(f"{name} Results - R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
                benchmark_results[name] = {"r2": float(r2), "rmse": float(rmse), "mae": float(mae)}
                
                # Log model run to mlflow
                self._init_mlflow(name, model)
                self._log_metrics_mlflow({"r2": float(r2), "rmse": float(rmse), "mae": float(mae)})
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_model_name = name
                    best_model = model
                    if preprocessor is not None:
                        best_pipeline = Pipeline(steps=[
                            ("preprocessor", preprocessor),
                            ("model", model),
                        ])
                    else:
                        best_pipeline = model
            except Exception as e:
                logger.error(f"Failed to train model {name}: {e}")

        logger.info(f"Best model selected: {best_model_name} with R2: {best_r2:.4f}")
        
        # Save benchmark results to artifacts
        benchmark_path = os.path.join(self.config.root_dir, "benchmark_results.json")
        with open(benchmark_path, "w") as f:
            json.dump(benchmark_results, f, indent=2)

        version_id = get_version_id()
        model_filename = f"model_{version_id}.joblib"
        model_path_str = os.path.join(self.config.root_dir, model_filename)
        
        try:
            with tempfile.NamedTemporaryFile(dir=self.config.root_dir, suffix='.joblib', delete=False) as tmp:
                tmp_path = tmp.name
                joblib.dump(best_pipeline, tmp_path)
            os.replace(tmp_path, model_path_str)
            checksum_path = model_path_str + ".sha256"
            from mlProject.utils.common import save_checksum
            save_checksum(Path(model_path_str), Path(checksum_path))
        except Exception as e:
            logger.exception(f"Failed to save model to {model_path_str}")
            raise

        model_path = Path(model_path_str)
        data_hash = None
        try:
            data_hash = compute_file_hash(Path(self.config.train_data_path))
        except Exception as e:
            logger.warning(f"Could not compute data hash: {e}")

        model_info = {
            "version_id": version_id,
            "model_path": str(model_path),
            "model_type": best_model_name,
            "params": {"alpha": self.config.alpha, "l1_ratio": self.config.l1_ratio} if best_model_name == "ElasticNet" else {},
            "data_hash": data_hash or "",
            "r2_score": best_r2
        }
        model_info_path = os.path.join(self.config.root_dir, "model_info.json")
        with open(model_info_path, "w") as f:
            json.dump(model_info, f, indent=2)

        # Log final promoted model to MLflow
        self._init_mlflow(f"FinalPromoted_{best_model_name}", best_model)
        self._log_to_mlflow(best_pipeline, version_id, model_path, model_info["params"], train_x)

        logger.info(f"Promoted pipeline ({best_model_name}) {version_id} saved to {model_path}")

    def _init_mlflow(self, model_name: str, model):
        try:
            from mlProject.config.configuration import ConfigurationManager
            config_manager = ConfigurationManager()
            registry_config = config_manager.get_model_registry_config()
            if registry_config.use_mlflow:
                self.mlflow_tracker = MlflowTracker(
                    tracking_uri=registry_config.mlflow_tracking_uri,
                    experiment_name=registry_config.mlflow_experiment_name,
                    use_mlflow=True,
                    registry_uri=registry_config.mlflow_registry_uri or None,
                )
                if self.mlflow_tracker.start_run(run_name=f"train_{model_name}"):
                    params = {"model_type": model_name}
                    if hasattr(model, "alpha"):
                        params["alpha"] = getattr(model, "alpha")
                    if hasattr(model, "l1_ratio"):
                        params["l1_ratio"] = getattr(model, "l1_ratio")
                    if hasattr(model, "n_estimators"):
                        params["n_estimators"] = getattr(model, "n_estimators")
                    if hasattr(model, "max_depth"):
                        params["max_depth"] = getattr(model, "max_depth")
                    self.mlflow_tracker.log_params(params)
        except Exception as e:
            logger.warning(f"Failed to initialize MLflow: {e}")
            self.mlflow_tracker = None

    def _log_metrics_mlflow(self, metrics: dict):
        if not self.mlflow_tracker or not self.mlflow_tracker.active_run:
            return
        try:
            self.mlflow_tracker.log_metrics(metrics)
        except Exception as e:
            logger.warning(f"Failed to log metrics to MLflow: {e}")
        finally:
            self.mlflow_tracker.end_run()

    def _log_to_mlflow(self, model, version_id, model_path, params, train_x):
        if not self.mlflow_tracker or not self.mlflow_tracker.active_run:
            return
        try:
            from mlflow.models import infer_signature
            signature = infer_signature(train_x, model.predict(train_x))
            self.mlflow_tracker.log_model(
                model=model,
                artifact_path="model",
                signature=signature,
                input_example=train_x.iloc[:5] if hasattr(train_x, 'iloc') else train_x[:5],
                registered_model_name=None,
            )
            self.mlflow_tracker.log_artifact(str(model_path), artifact_path="artifacts")
            preprocessor_path = self.config.preprocessor_path
            if preprocessor_path and Path(preprocessor_path).exists():
                self.mlflow_tracker.log_artifact(str(preprocessor_path), artifact_path="artifacts")
            logger.info(f"Model {version_id} logged to MLflow")
        except Exception as e:
            logger.warning(f"Failed to log model to MLflow: {e}")
        finally:
            self.mlflow_tracker.end_run()
