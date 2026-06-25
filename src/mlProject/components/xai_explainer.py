import os
import shap
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline as SklearnPipeline
from mlProject import logger
from mlProject.components.data_transformation import NUMERIC_FEATURES

_TREE_ESTIMATOR_TYPES = frozenset([
    "RandomForestRegressor", "GradientBoostingRegressor",
    "XGBRegressor", "ExtraTreesRegressor", "DecisionTreeRegressor",
    "RandomForestClassifier", "GradientBoostingClassifier",
    "XGBClassifier", "ExtraTreesClassifier", "DecisionTreeClassifier",
])

_LINEAR_ESTIMATOR_TYPES = frozenset([
    "ElasticNet", "Ridge", "Lasso", "LinearRegression",
    "LogisticRegression", "SGDRegressor", "SGDClassifier",
])


class XAIExplainer:
    def __init__(self, model, training_data_path: str = "artifacts/data_transformation/train.csv"):
        self.model = model
        self.training_data_path = training_data_path
        self.explainer = None
        self._preprocessor = None
        self._init_explainer()

    def _split_pipeline(self):
        """Return (preprocessor, estimator). Preprocessor is None for non-Pipeline models."""
        if isinstance(self.model, SklearnPipeline) and len(self.model.steps) > 1:
            preprocessor = SklearnPipeline(self.model.steps[:-1])
            estimator = self.model.steps[-1][1]
            return preprocessor, estimator
        return None, self.model

    def _transform_features(self, df: pd.DataFrame) -> np.ndarray:
        if self._preprocessor is not None:
            return self._preprocessor.transform(df)
        return df.values

    def _init_explainer(self):
        try:
            if os.path.exists(self.training_data_path):
                train_df = pd.read_csv(self.training_data_path)
                if "quality" in train_df.columns:
                    train_df = train_df.drop(columns=["quality"])
                train_df = train_df[NUMERIC_FEATURES]
                background_data = train_df.values
            else:
                background_data = None

            preprocessor, estimator = self._split_pipeline()
            self._preprocessor = preprocessor
            est_name = type(estimator).__name__

            if background_data is not None and len(background_data) > 100:
                idx = np.random.default_rng(42).choice(len(background_data), 100, replace=False)
                background_sample = background_data[idx]
            else:
                background_sample = background_data

            if preprocessor is not None and background_sample is not None:
                background_transformed = preprocessor.transform(
                    pd.DataFrame(background_sample, columns=NUMERIC_FEATURES)
                )
            else:
                background_transformed = background_sample

            if est_name in _TREE_ESTIMATOR_TYPES:
                self.explainer = shap.TreeExplainer(estimator)
            elif est_name in _LINEAR_ESTIMATOR_TYPES:
                if background_transformed is not None:
                    self.explainer = shap.LinearExplainer(estimator, background_transformed)
                else:
                    dummy = np.zeros((1, estimator.coef_.shape[0] if hasattr(estimator, "coef_") else len(NUMERIC_FEATURES)))
                    self.explainer = shap.LinearExplainer(estimator, dummy)
            else:
                if background_data is not None:
                    bg = shap.kmeans(background_data, 10)
                    self.explainer = shap.Explainer(self.model.predict, bg)
                else:
                    dummy = np.zeros((1, len(NUMERIC_FEATURES)))
                    self.explainer = shap.Explainer(self.model.predict, dummy)

            logger.info(f"SHAP explainer initialized: {type(self.explainer).__name__} for {est_name}")
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            self.explainer = None

    def explain_instance(self, features_dict: dict) -> dict:
        """
        Explain a single prediction.
        features_dict: dictionary of feature names and values.
        """
        if self.explainer is None:
            self._init_explainer()
            if self.explainer is None:
                raise RuntimeError("SHAP Explainer could not be initialized.")

        df = pd.DataFrame([features_dict])[NUMERIC_FEATURES]

        try:
            input_data = self._transform_features(df)
            explanation = self.explainer(input_data)
            shap_values = explanation.values[0].tolist()
            base_value = float(explanation.base_values[0] if hasattr(explanation.base_values, "__len__") else explanation.base_values)

            contributions = []
            for col, val, shap_val in zip(NUMERIC_FEATURES, df.values[0], shap_values):
                contributions.append({
                    "feature": col,
                    "value": float(val),
                    "shap_value": float(shap_val)
                })

            prediction = float(self.model.predict(df.values)[0])

            return {
                "base_value": base_value,
                "prediction": prediction,
                "contributions": contributions
            }
        except Exception as e:
            logger.error(f"Error computing local SHAP values: {e}")
            try:
                model_step = self.model
                if hasattr(self.model, "steps"):
                    model_step = self.model.steps[-1][1]
                if hasattr(model_step, "coef_"):
                    coefs = model_step.coef_
                    contributions = []
                    for col, val, coef in zip(NUMERIC_FEATURES, df.values[0], coefs):
                        contributions.append({
                            "feature": col,
                            "value": float(val),
                            "shap_value": float(coef * val)
                        })
                    return {
                        "base_value": float(model_step.intercept_),
                        "prediction": float(self.model.predict(df.values)[0]),
                        "contributions": contributions,
                        "fallback": True
                    }
            except Exception as ex:
                logger.error(f"XAI fallback also failed: {ex}")
            raise e

    def get_global_importance(self) -> dict:
        """
        Compute global feature importance (mean absolute SHAP values) using training data.
        """
        if not os.path.exists(self.training_data_path):
            return {"error": "Training data not found. Run training first."}

        try:
            train_df = pd.read_csv(self.training_data_path)
            if "quality" in train_df.columns:
                train_df = train_df.drop(columns=["quality"])
            train_df = train_df[NUMERIC_FEATURES]

            if self.explainer is None:
                self._init_explainer()
                if self.explainer is None:
                    return {"error": "SHAP Explainer could not be initialized."}

            sample_df = train_df.sample(min(100, len(train_df)), random_state=42)
            input_data = self._transform_features(sample_df)
            explanation = self.explainer(input_data)
            mean_abs_shap = np.mean(np.abs(explanation.values), axis=0)

            importances = []
            for col, imp in zip(NUMERIC_FEATURES, mean_abs_shap):
                importances.append({
                    "feature": col,
                    "importance": float(imp)
                })

            importances = sorted(importances, key=lambda x: x["importance"], reverse=True)
            return {"importances": importances}
        except Exception as e:
            logger.error(f"Error computing global SHAP values: {e}")
            return {"error": str(e)}
