import os
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature

from mlProject import logger


class MlflowTracker:
    def __init__(
        self,
        tracking_uri: str = "./mlruns",
        experiment_name: str = "wine_quality_prediction",
        use_mlflow: bool = False,
        registry_uri: Optional[str] = None,
    ):
        self.use_mlflow = use_mlflow
        if not use_mlflow:
            return

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        tracking_uri = mlflow.get_tracking_uri()
        parsed = urlparse(tracking_uri)
        self._is_local = parsed.scheme == "" or parsed.scheme == "file"

        try:
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            logger.warning(f"Failed to set MLflow experiment '{experiment_name}': {e}")

        if registry_uri:
            self._client = MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)
        else:
            self._client = MlflowClient(tracking_uri=tracking_uri)

        self._active_run = None

    @property
    def active_run(self):
        return self._active_run

    def start_run(self, run_name: Optional[str] = None) -> bool:
        if not self.use_mlflow:
            return False
        try:
            self._active_run = mlflow.start_run(run_name=run_name)
            logger.info(f"MLflow run started: {self._active_run.info.run_id}")
            return True
        except Exception as e:
            logger.warning(f"Failed to start MLflow run: {e}")
            return False

    def end_run(self):
        if self._active_run:
            try:
                mlflow.end_run()
                logger.info(f"MLflow run ended: {self._active_run.info.run_id}")
            except Exception as e:
                logger.warning(f"Failed to end MLflow run: {e}")
            self._active_run = None

    def log_params(self, params: dict) -> bool:
        if not self.use_mlflow or not self._active_run:
            return False
        try:
            mlflow.log_params(params)
            return True
        except Exception as e:
            logger.warning(f"Failed to log params to MLflow: {e}")
            return False

    def log_metrics(self, metrics: dict, step: Optional[int] = None) -> bool:
        if not self.use_mlflow or not self._active_run:
            return False
        try:
            mlflow.log_metrics(metrics, step=step)
            return True
        except Exception as e:
            logger.warning(f"Failed to log metrics to MLflow: {e}")
            return False

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> bool:
        if not self.use_mlflow or not self._active_run:
            return False
        try:
            mlflow.log_artifact(local_path, artifact_path=artifact_path)
            return True
        except Exception as e:
            logger.warning(f"Failed to log artifact to MLflow: {e}")
            return False

    def log_model(
        self,
        model,
        artifact_path: str = "model",
        signature=None,
        input_example=None,
        registered_model_name: Optional[str] = None,
        pip_requirements: Optional[list] = None,
    ) -> bool:
        if not self.use_mlflow or not self._active_run:
            return False
        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_path,
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_model_name,
                pip_requirements=pip_requirements,
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to log model to MLflow: {e}")
            return False

    def register_model_version(
        self,
        model_name: str,
        run_id: Optional[str] = None,
        source: Optional[str] = None,
    ) -> Optional[str]:
        if not self.use_mlflow:
            return None
        try:
            if not run_id and self._active_run:
                run_id = self._active_run.info.run_id
            if not run_id:
                logger.warning("No run_id available for model registration")
                return None

            result = mlflow.register_model(
                model_uri=f"runs:/{run_id}/{source or 'model'}",
                name=model_name,
            )
            version = result.version
            logger.info(f"Model registered in MLflow Registry: {model_name} v{version}")
            return version
        except Exception as e:
            logger.warning(f"Failed to register model in MLflow Registry: {e}")
            return None

    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str = "Staging",
    ) -> bool:
        if not self.use_mlflow:
            return False
        try:
            self._client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
            )
            logger.info(f"Model {model_name} v{version} transitioned to {stage}")
            return True
        except Exception as e:
            logger.warning(f"Failed to transition model stage in MLflow: {e}")
            return False

    def get_tracking_ui_url(self) -> Optional[str]:
        if not self.use_mlflow:
            return None
        if self._is_local:
            return "http://localhost:5000"
        tracking_uri = mlflow.get_tracking_uri()
        return tracking_uri

    def get_experiment_url(self) -> Optional[str]:
        if not self.use_mlflow or not self._active_run:
            return None
        try:
            experiment_id = self._active_run.info.experiment_id
            tracking_uri = mlflow.get_tracking_uri()
            if self._is_local:
                return f"http://localhost:5000/#/experiments/{experiment_id}"
            return f"{tracking_uri}/#/experiments/{experiment_id}"
        except Exception:
            return None

    def cleanup(self):
        self.end_run()
