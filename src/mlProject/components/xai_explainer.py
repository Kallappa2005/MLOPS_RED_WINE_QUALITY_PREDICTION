import os
import shap
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from mlProject import logger
from mlProject.components.data_transformation import NUMERIC_FEATURES

class XAIExplainer:
    def __init__(self, model, training_data_path: str = "artifacts/data_transformation/train.csv"):
        self.model = model
        self.training_data_path = training_data_path
        self.explainer = None
        self._init_explainer()
        
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
                
            if background_data is not None:
                if len(background_data) > 100:
                    background_data = shap.kmeans(background_data, 10)
                self.explainer = shap.Explainer(self.model.predict, background_data)
            else:
                dummy = np.zeros((1, len(NUMERIC_FEATURES)))
                self.explainer = shap.Explainer(self.model.predict, dummy)
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
            explanation = self.explainer(df)
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
            explanation = self.explainer(sample_df)
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
