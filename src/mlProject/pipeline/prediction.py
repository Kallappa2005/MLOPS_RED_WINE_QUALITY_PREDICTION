import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from mlProject.components.data_transformation import NUMERIC_FEATURES
from mlProject.config.configuration import ConfigurationManager
from mlProject.utils.common import load_env_file
from mlProject.utils.model_registry import get_production_model_path, load_registry
from mlProject import logger

class PredictionPipeline:
    def __init__(self, model_path: Path = None):
        self.unified_pipeline = None
        self.explainer = None
        self._model_path = model_path
        self._loaded_mtime = None
        if model_path is None:
            load_env_file()
            try:
                config_manager = ConfigurationManager()
                # Prefer the stable model.joblib: it is rewritten on both
                # promotion and rollback, so its mtime change drives the
                # hot-reload in predict() and live traffic follows rollbacks.
                model_eval_config = config_manager.get_model_evaluation_config()
                stable_path = model_eval_config.model_path
                if stable_path is not None and stable_path.exists():
                    self._model_path = stable_path
                else:
                    registry_config = config_manager.get_model_registry_config()
                    prod_path = get_production_model_path(registry_config.registry_path)
                    self._model_path = prod_path if prod_path is not None else stable_path
            except Exception:
                self._model_path = Path('artifacts/model_trainer/model.joblib')

        # Verify loaded model matches registry production version
        if self._model_path is not None and self._model_path.name == "model.joblib":
            try:
                model_info_path = self._model_path.parent / "model_info.json"
                if model_info_path.exists():
                    with open(model_info_path) as f:
                        model_info = json.load(f)
                    loaded_version = model_info.get("version_id")
                    if loaded_version:
                        cm = ConfigurationManager()
                        reg_cfg = cm.get_model_registry_config()
                        registry = load_registry(reg_cfg.registry_path)
                        prod_version = registry.get("production")
                        if prod_version and loaded_version != prod_version:
                            logger.critical(
                                f"Loaded model version {loaded_version} does not match "
                                f"registry production version {prod_version}. "
                                f"Predictions may be stale."
                            )
            except Exception:
                pass

    def predict(self, data):
        model_path = self._model_path or Path('artifacts/model_trainer/model.joblib')
        current_mtime = model_path.stat().st_mtime if model_path.exists() else None
        if self.unified_pipeline is None or (current_mtime and self._loaded_mtime != current_mtime):
            from mlProject.utils.common import verify_model_integrity
            checksum_path = Path(str(model_path) + ".sha256")
            if not verify_model_integrity(model_path, checksum_path):
                raise ValueError(f"Model integrity check failed for {model_path}")
            self.unified_pipeline = joblib.load(model_path)
            self._loaded_mtime = current_mtime
            logger.info(f"Loaded unified pipeline from {model_path}")
            
            # Load SHAP explainer
            explainer_path = model_path.parent / "explainer.joblib"
            if explainer_path.exists():
                try:
                    self.explainer = joblib.load(explainer_path)
                    logger.info(f"Loaded SHAP explainer from {explainer_path}")
                except Exception as e:
                    logger.warning(f"Failed to load SHAP explainer: {e}")
                    self.explainer = None
            else:
                self.explainer = None

        if isinstance(data, np.ndarray):
            if data.shape[1] != len(NUMERIC_FEATURES):
                raise ValueError(
                    f"Expected {len(NUMERIC_FEATURES)} features, got {data.shape[1]}. "
                    f"Required columns in order: {NUMERIC_FEATURES}"
                )
            input_data = pd.DataFrame(data, columns=NUMERIC_FEATURES)
        elif isinstance(data, pd.DataFrame):
            missing = [col for col in NUMERIC_FEATURES if col not in data.columns]
            if missing:
                raise ValueError(
                    f"Missing required feature columns: {missing}. "
                    f"Expected columns: {NUMERIC_FEATURES}"
                )
            extra = [col for col in data.columns if col not in NUMERIC_FEATURES]
            if extra:
                logger.warning(f"Extra columns ignored during prediction: {extra}")
            input_data = data[NUMERIC_FEATURES].values
        else:
            input_data = data

        # Unified pipeline handles preprocessing (if available) and prediction
        try:
            prediction = self.unified_pipeline.predict(input_data)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise RuntimeError(f"Model prediction failed: {e}") from e
        
        return prediction

    def explain(self, data):
        """Returns SHAP values for the given data if explainer is loaded."""
        if self.explainer is None:
            return None
            
        if isinstance(data, np.ndarray):
            input_data = data
        elif isinstance(data, pd.DataFrame):
            input_data = data[NUMERIC_FEATURES].values
        else:
            input_data = data
            
        try:
            shap_values = self.explainer(input_data)
            # Ensure feature names are attached for plot labeling
            if getattr(shap_values, "feature_names", None) is None:
                shap_values.feature_names = NUMERIC_FEATURES
            return shap_values
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return None
