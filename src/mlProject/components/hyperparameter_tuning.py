import os
import json
import joblib
import pandas as pd
import optuna
import mlflow
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
from mlProject import logger
from mlProject.entity.config_entity import HyperparameterTuningConfig
from mlProject.components.data_transformation import NUMERIC_FEATURES

class HyperparameterTuner:
    def __init__(self, config: HyperparameterTuningConfig):
        self.config = config

    def tune(self):
        try:
            train_data = pd.read_csv(self.config.train_data_path)
        except Exception as e:
            logger.exception("Failed to load training data for tuning")
            raise

        train_x = train_data.drop([self.config.target_column], axis=1)
        train_y = train_data[self.config.target_column]

        if self.config.use_scaler and self.config.preprocessor_path and self.config.preprocessor_path.exists():
            preprocessor = joblib.load(self.config.preprocessor_path)
            # Ensure correct columns
            expected_cols = len(NUMERIC_FEATURES)
            if train_x.shape[1] != expected_cols:
                train_x = train_x[NUMERIC_FEATURES]
            train_x_preprocessed = preprocessor.transform(train_x)
        else:
            train_x_preprocessed = train_x.values

        def objective(trial):
            alpha = trial.suggest_float("alpha", 0.001, 2.0, log=True)
            l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)

            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
            # Minimize RMSE (since cross_val_score returns negative RMSE)
            scores = cross_val_score(model, train_x_preprocessed, train_y, cv=5, scoring='neg_root_mean_squared_error')
            rmse = -1 * scores.mean()
            return rmse

        logger.info(f"Starting hyperparameter tuning with {self.config.n_trials} trials")
        
        try:
            from optuna.integration.mlflow import MLflowCallback
            # By default it logs to the active MLflow experiment
            mlflow_callback = MLflowCallback(
                tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"),
                metric_name="rmse"
            )
            callbacks = [mlflow_callback]
        except ImportError:
            logger.warning("optuna.integration.mlflow not available, skipping MLflow callback")
            callbacks = []

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.config.n_trials, callbacks=callbacks)
        
        best_params = study.best_params
        logger.info(f"Tuning completed. Best parameters: {best_params}")
        logger.info(f"Best RMSE: {study.best_value}")

        output_path = self.config.root_dir / "best_params.json"
        with open(output_path, "w") as f:
            json.dump(best_params, f, indent=4)
        logger.info(f"Saved best parameters to {output_path}")

