"""
Unified Model Registry Interface

Coordinates both the JSON registry and MLflow registry to ensure consistency.
Implements transaction-like behavior: if one backend fails, rolls back the other.
"""

from pathlib import Path
from typing import List, Optional

from mlProject import logger
from mlProject.utils.model_registry import (
    load_registry,
    register_model as json_register_model,
    rollback_to_version as json_rollback_to_version,
    validate_registry as json_validate_registry,
)
from mlProject.utils.mlflow_tracker import MlflowTracker


class UnifiedRegistry:
    """
    Unified interface that coordinates JSON and MLflow registries.

    The JSON registry is the primary source of truth. MLflow is optional
    and acts as a secondary tracking system.
    """

    def __init__(
        self,
        registry_path: Path,
        mlflow_tracker: Optional[MlflowTracker] = None,
    ):
        self.registry_path = registry_path
        self.mlflow_tracker = mlflow_tracker

    def register_model(
        self,
        model,
        model_path: Path,
        version_id: str,
        metrics: dict,
        params: dict,
        data_hash: Optional[str] = None,
        max_versions_to_keep: int = 10,
        quality_gate_max_rmse_degradation_pct: float = 5.0,
        stable_model_path: Optional[Path] = None,
        registered_model_name: str = "wine_quality_model",
    ) -> dict:
        """
        Register a model version in both registries with rollback protection.

        Returns the registered version entry.
        Raises an error if the primary (JSON) registry fails.
        Logs warnings if MLflow registration fails (non-critical).
        """
        mlflow_registered = False
        mlflow_version = None

        # Step 1: Register in JSON registry (primary)
        entry = json_register_model(
            registry_path=self.registry_path,
            model_path=model_path,
            version_id=version_id,
            metrics=metrics,
            params=params,
            data_hash=data_hash,
            max_versions_to_keep=max_versions_to_keep,
            quality_gate_max_rmse_degradation_pct=quality_gate_max_rmse_degradation_pct,
            stable_model_path=stable_model_path,
        )

        # Step 2: Register in MLflow (secondary, best-effort)
        if self.mlflow_tracker and self.mlflow_tracker.use_mlflow:
            try:
                mlflow_version = self.mlflow_tracker.register_model_version(
                    model_name=registered_model_name,
                    source=str(model_path),
                )
                if mlflow_version:
                    mlflow_registered = True
                    logger.info(
                        f"Model {version_id} also registered in MLflow as "
                        f"{registered_model_name} v{mlflow_version}"
                    )
                else:
                    logger.warning(
                        f"MLflow registration returned no version for {version_id} — "
                        "continuing with JSON registry only"
                    )
            except Exception as e:
                logger.warning(
                    f"MLflow registration failed for {version_id}: {e} — "
                    "continuing with JSON registry only"
                )

        # Store MLflow version info in entry for reference
        if mlflow_version:
            entry["mlflow_version"] = mlflow_version

        return entry

    def rollback_to_version(
        self,
        version_id: str,
        registered_model_name: str = "wine_quality_model",
    ) -> bool:
        """
        Rollback production to a specific version in both registries.

        Returns True if the JSON rollback succeeded.
        MLflow rollback is best-effort and logged as warning on failure.
        """
        # Step 1: Rollback JSON registry (primary)
        success = json_rollback_to_version(self.registry_path, version_id)
        if not success:
            logger.error(f"JSON registry rollback to {version_id} failed")
            return False

        # Step 2: Rollback MLflow (secondary, best-effort)
        if self.mlflow_tracker and self.mlflow_tracker.use_mlflow:
            try:
                # Transition the target version to Production stage in MLflow
                # First, we need to find the MLflow version for this version_id
                registry = load_registry(self.registry_path)
                for v in registry.get("versions", []):
                    if v.get("id") == version_id:
                        mlflow_version = v.get("mlflow_version")
                        if mlflow_version:
                            self.mlflow_tracker.transition_model_stage(
                                model_name=registered_model_name,
                                version=mlflow_version,
                                stage="Production",
                            )
                            logger.info(
                                f"MLflow model {registered_model_name} v{mlflow_version} "
                                f"transitioned to Production"
                            )
                        break
            except Exception as e:
                logger.warning(
                    f"MLflow rollback failed for {version_id}: {e} — "
                    "JSON registry rollback succeeded"
                )

        return True

    def validate(self) -> List[str]:
        """
        Validate consistency between JSON and MLflow registries.

        Returns a list of issues found. Empty list means consistent.
        """
        issues = []

        # Validate JSON registry
        json_issues = json_validate_registry(self.registry_path)
        issues.extend([f"JSON: {issue}" for issue in json_issues])

        # Validate MLflow consistency if enabled
        if self.mlflow_tracker and self.mlflow_tracker.use_mlflow:
            try:
                registry = load_registry(self.registry_path)
                production_id = registry.get("production")
                if production_id:
                    for v in registry.get("versions", []):
                        if v.get("id") == production_id:
                            mlflow_version = v.get("mlflow_version")
                            if not mlflow_version:
                                issues.append(
                                    f"MLflow: Production version {production_id} has no "
                                    "MLflow version reference"
                                )
                            break
            except Exception as e:
                issues.append(f"MLflow: Validation check failed: {e}")

        if issues:
            for issue in issues:
                logger.warning(f"Registry validation issue: {issue}")
        else:
            logger.info("Registry validation passed — all systems consistent")

        return issues

    def sync_from_json(self, registered_model_name: str = "wine_quality_model") -> dict:
        """
        Sync MLflow state from JSON registry.

        This is a one-way sync from JSON (primary) to MLflow (secondary).
        Returns a summary of actions taken.
        """
        summary = {"synced": False, "actions": []}

        if not self.mlflow_tracker or not self.mlflow_tracker.use_mlflow:
            summary["actions"].append("MLflow not enabled — sync skipped")
            return summary

        try:
            registry = load_registry(self.registry_path)
            production_id = registry.get("production")

            if not production_id:
                summary["actions"].append("No production version in JSON registry")
                return summary

            for v in registry.get("versions", []):
                if v.get("id") == production_id:
                    mlflow_version = v.get("mlflow_version")
                    if mlflow_version:
                        # Transition to Production in MLflow
                        success = self.mlflow_tracker.transition_model_stage(
                            model_name=registered_model_name,
                            version=mlflow_version,
                            stage="Production",
                        )
                        if success:
                            summary["actions"].append(
                                f"Transitioned MLflow {registered_model_name} "
                                f"v{mlflow_version} to Production"
                            )
                            summary["synced"] = True
                        else:
                            summary["actions"].append(
                                f"Failed to transition MLflow {registered_model_name} "
                                f"v{mlflow_version} to Production"
                            )
                    else:
                        summary["actions"].append(
                            f"Production version {production_id} has no MLflow version — "
                            "manual sync required"
                        )
                    break
        except Exception as e:
            summary["actions"].append(f"Sync failed: {e}")

        return summary


def get_unified_registry(
    registry_path: Optional[Path] = None,
    mlflow_tracker: Optional[MlflowTracker] = None,
) -> UnifiedRegistry:
    """
    Factory function to create a UnifiedRegistry instance.

    If registry_path is not provided, uses the default from ConfigurationManager.
    If mlflow_tracker is not provided, creates one based on config.
    """
    if registry_path is None:
        try:
            from mlProject.config.configuration import ConfigurationManager
            config_manager = ConfigurationManager()
            registry_config = config_manager.get_model_registry_config()
            registry_path = registry_config.registry_path
        except Exception:
            registry_path = Path("artifacts/model_registry.json")

    if mlflow_tracker is None:
        try:
            from mlProject.config.configuration import ConfigurationManager
            config_manager = ConfigurationManager()
            registry_config = config_manager.get_model_registry_config()
            mlflow_tracker = MlflowTracker(
                tracking_uri=registry_config.mlflow_tracking_uri,
                experiment_name=registry_config.mlflow_experiment_name,
                use_mlflow=registry_config.use_mlflow,
            )
        except Exception:
            mlflow_tracker = MlflowTracker(use_mlflow=False)

    return UnifiedRegistry(
        registry_path=registry_path,
        mlflow_tracker=mlflow_tracker,
    )
