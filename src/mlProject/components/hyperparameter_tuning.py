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

        try:
            from optuna.integration.mlflow import MLflowCallback
            # By default it logs to the active MLflow experiment
            mlflow_callback = MLflowCallback(
                tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"),
                metric_name="rmse"
            )
            callbacks = [mlflow_callback]
        except ImportError:
            mlflow_callback = None
            logger.warning("optuna.integration.mlflow not available, skipping MLflow callback")
            callbacks = []

        def objective(trial):
            alpha = trial.suggest_float("alpha", self.config.alpha_min, self.config.alpha_max, log=True)
            l1_ratio = trial.suggest_float("l1_ratio", self.config.l1_ratio_min, self.config.l1_ratio_max)

            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
            
            from sklearn.model_selection import cross_validate
            scoring = {
                'rmse': 'neg_root_mean_squared_error',
                'r2': 'r2'
            }
            # Calculate RMSE and R2 via cross-validation
            cv_results = cross_validate(model, train_x_preprocessed, train_y, cv=5, scoring=scoring)
            
            rmse = -1 * cv_results['test_rmse'].mean()
            r2 = cv_results['test_r2'].mean()
            
            if mlflow.active_run():
                mlflow.log_metric("r2", r2)
                
            return rmse

        if mlflow_callback:
            objective = mlflow_callback.track_in_mlflow()(objective)

        logger.info(f"Starting hyperparameter tuning with {self.config.n_trials} trials")

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.config.n_trials, callbacks=callbacks)
        
        best_params = study.best_params
        logger.info(f"Tuning completed. Best parameters: {best_params}")
        logger.info(f"Best RMSE: {study.best_value}")

        output_path = self.config.root_dir / "best_params.json"
        with open(output_path, "w") as f:
            json.dump(best_params, f, indent=4)
        logger.info(f"Saved best parameters to {output_path}")

