import os
from pathlib import Path
from mlProject.constants import *
from mlProject import logger
from mlProject.utils.common import read_yaml, create_directories, get_env_or_config, load_env_file
from mlProject.entity.config_entity import (DataIngestionConfig,
                                            DataValidationConfig,
                                            DataTransformationConfig,
                                            HyperparameterTuningConfig,
                                            ModelTrainerConfig,
                                            ModelEvaluationConfig,
                                            ModelRegistryConfig)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH,
        schema_filepath: Path = SCHEMA_FILE_PATH):

        load_env_file()
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        artifacts_root = get_env_or_config(ENV_ARTIFACTS_ROOT, self.config.artifacts_root)
        create_directories([artifacts_root])


    def _validate_config_keys(self, config_section: dict, required_keys: list, section_name: str):
        missing = [key for key in required_keys if key not in config_section]
        if missing:
            raise KeyError(
                f"Missing required config keys in '{section_name}': {missing}"
            )

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        self._validate_config_keys(config, ["root_dir", "source_URL", "local_data_file", "unzip_dir"], "data_ingestion")

        root_dir = get_env_or_config(ENV_DATA_INGESTION_ROOT_DIR, config.root_dir)
        source_URL = get_env_or_config(ENV_DATA_INGESTION_SOURCE_URL, config.source_URL)

        create_directories([root_dir])

        expected_checksum = get_env_or_config(ENV_DATA_INGESTION_EXPECTED_CHECKSUM, config.get("expected_checksum", ""))

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(root_dir),
            source_URL=source_URL,
            local_data_file=Path(get_env_or_config(ENV_DATA_INGESTION_LOCAL_DATA_FILE, config.local_data_file)),
            unzip_dir=Path(get_env_or_config(ENV_DATA_INGESTION_UNZIP_DIR, config.unzip_dir)),
            expected_checksum=expected_checksum
        )

        return data_ingestion_config
    

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS
        self._validate_config_keys(config, ["root_dir", "STATUS_FILE", "data_file"], "data_validation")

        root_dir = get_env_or_config(ENV_DATA_VALIDATION_ROOT_DIR, config.root_dir)
        create_directories([root_dir])

        drift_threshold = float(get_env_or_config(
            ENV_DATA_VALIDATION_DRIFT_THRESHOLD,
            self.params.get("DataValidation", {}).get("drift_threshold", 0.05),
            transform=float
        ))

        reference_data_path = get_env_or_config(
            ENV_DATA_VALIDATION_REFERENCE_DATA_PATH,
            config.get("reference_data_path", "artifacts/reference_data.csv"),
        )

        data_validation_config = DataValidationConfig(
            root_dir=Path(root_dir),
            STATUS_FILE=Path(get_env_or_config(ENV_DATA_VALIDATION_STATUS_FILE, config.STATUS_FILE)),
            data_file=Path(get_env_or_config(ENV_DATA_VALIDATION_DATA_FILE, config.data_file)),
            all_schema=schema,
            drift_threshold=drift_threshold,
            reference_data_path=Path(reference_data_path),
        )

        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        params = self.params.DataTransformation
        preproc = self.params.get("Preprocessing", {})
        self._validate_config_keys(config, ["root_dir", "data_path"], "data_transformation")

        root_dir = get_env_or_config(ENV_DATA_TRANSFORMATION_ROOT_DIR, config.root_dir)
        create_directories([root_dir])

        use_scaler = os.environ.get(ENV_USE_SCALER, "true").lower() in ("1", "true", "yes")
        scaler_type = os.environ.get(ENV_SCALER_TYPE, params.get("feature_scaling", {}).get("method", "standard"))

        preprocessor_path = Path(config.get("preprocessor_path", str(Path(config.root_dir) / "preprocessor.joblib")))

        min_samples_per_class = int(params.get("min_samples_per_class", 4))
        enable_per_class_evaluation = str(params.get("enable_per_class_evaluation", "true")).lower() in ("1", "true", "yes")

        data_transformation_config = DataTransformationConfig(
            root_dir=Path(root_dir),
            data_path=Path(get_env_or_config(ENV_DATA_TRANSFORMATION_DATA_PATH, config.data_path)),
            test_size=params.test_size,
            random_state=params.random_state,
            stratify_column=params.stratify_column,
            use_scaler=use_scaler,
            scaler_type=scaler_type,
            handle_outliers=preproc.get("handle_outliers", True),
            outlier_method=preproc.get("outlier_method", "iqr"),
            outlier_iqr_multiplier=preproc.get("outlier_iqr_multiplier", 1.5),
            impute_missing=preproc.get("impute_missing", True),
            feature_engineering_flags=preproc.get("feature_engineering_flags", None),
            preprocessor_path=Path(preprocessor_path),
            min_samples_per_class=min_samples_per_class,
            enable_per_class_evaluation=enable_per_class_evaluation,
        )

        return data_transformation_config
    
    def get_hyperparameter_tuning_config(self) -> HyperparameterTuningConfig:
        config = self.config.hyperparameter_tuning
        schema = self.schema.TARGET_COLUMN
        tuning_params = self.params.get("HyperparameterTuning", {})
        self._validate_config_keys(config, ["root_dir"], "hyperparameter_tuning")

        root_dir = get_env_or_config("ENV_HYPERPARAMETER_TUNING_ROOT_DIR", config.root_dir)
        create_directories([root_dir])

        preprocessor_path = self.config.data_transformation.get("preprocessor_path")
        if preprocessor_path is None:
            preprocessor_path = str(Path(self.config.data_transformation.root_dir) / "preprocessor.joblib")

        n_trials = int(tuning_params.get("n_trials", config.get("n_trials", 20)))
        alpha_min = float(tuning_params.get("alpha_min", 0.001))
        alpha_max = float(tuning_params.get("alpha_max", 2.0))
        l1_ratio_min = float(tuning_params.get("l1_ratio_min", 0.0))
        l1_ratio_max = float(tuning_params.get("l1_ratio_max", 1.0))

        return HyperparameterTuningConfig(
            root_dir=Path(root_dir),
            train_data_path=Path(self.config.model_trainer.train_data_path),
            test_data_path=Path(self.config.model_trainer.test_data_path),
            target_column=schema.name,
            preprocessor_path=Path(preprocessor_path),
            use_scaler=self.params.Preprocessing.use_scaler,
            n_trials=n_trials,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            l1_ratio_min=l1_ratio_min,
            l1_ratio_max=l1_ratio_max,
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.ElasticNet
        schema = self.schema.TARGET_COLUMN
        self._validate_config_keys(config, ["root_dir", "train_data_path", "test_data_path", "model_name"], "model_trainer")

        root_dir = get_env_or_config(ENV_MODEL_TRAINER_ROOT_DIR, config.root_dir)
        create_directories([root_dir])

        preprocessor_path = self.config.data_transformation.get("preprocessor_path")
        if preprocessor_path is None:
            preprocessor_path = str(Path(self.config.data_transformation.root_dir) / "preprocessor.joblib")

        alpha = float(get_env_or_config("ENV_ELASTICNET_ALPHA", params.alpha, transform=float))
        l1_ratio = float(get_env_or_config("ENV_ELASTICNET_L1_RATIO", params.l1_ratio, transform=float))

        # Override with tuned params if they exist
        tuned_params_path = Path("artifacts/hyperparameter_tuning/best_params.json")
        if tuned_params_path.exists():
            try:
                import json
                with open(tuned_params_path, "r") as f:
                    best_params = json.load(f)
                if "alpha" in best_params:
                    alpha = float(best_params["alpha"])
                if "l1_ratio" in best_params:
                    l1_ratio = float(best_params["l1_ratio"])
                logger.info(f"Loaded tuned parameters from {tuned_params_path}: alpha={alpha}, l1_ratio={l1_ratio}")
            except Exception as e:
                logger.warning(f"Failed to load tuned parameters: {e}")

        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(root_dir),
            train_data_path=Path(get_env_or_config(ENV_MODEL_TRAINER_TRAIN_DATA_PATH, config.train_data_path)),
            test_data_path=Path(get_env_or_config(ENV_MODEL_TRAINER_TEST_DATA_PATH, config.test_data_path)),
            model_name=get_env_or_config(ENV_MODEL_TRAINER_MODEL_NAME, config.model_name),
            alpha=alpha,
            l1_ratio=l1_ratio,
            target_column=schema.name,
            preprocessor_path=Path(preprocessor_path),
            use_scaler=self.params.get("Preprocessing", {}).get("use_scaler", True),
        )

        return model_trainer_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params.ElasticNet
        schema = self.schema.TARGET_COLUMN
        self._validate_config_keys(config, ["root_dir", "test_data_path", "model_path", "metric_file_name"], "model_evaluation")

        root_dir = get_env_or_config(ENV_MODEL_EVALUATION_ROOT_DIR, config.root_dir)
        create_directories([root_dir])

        preprocessor_path = self.config.data_transformation.get("preprocessor_path")
        if preprocessor_path is None:
            preprocessor_path = str(Path(self.config.data_transformation.root_dir) / "preprocessor.joblib")

        eval_params = self.params.get("ModelEvaluation", {})
        per_class_r2_threshold = float(eval_params.get("per_class_r2_threshold", -0.5))

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=Path(root_dir),
            test_data_path=Path(get_env_or_config(ENV_MODEL_EVALUATION_TEST_DATA_PATH, config.test_data_path)),
            model_path=Path(get_env_or_config(ENV_MODEL_EVALUATION_MODEL_PATH, config.model_path)),
            all_params=params,
            metric_file_name=Path(get_env_or_config(ENV_MODEL_EVALUATION_METRIC_FILE_NAME, config.metric_file_name)),
            target_column=schema.name,
            preprocessor_path=Path(preprocessor_path),
            use_scaler=self.params.get("Preprocessing", {}).get("use_scaler", True),
            per_class_r2_threshold=per_class_r2_threshold,
        )

        return model_evaluation_config

    def get_model_registry_config(self) -> ModelRegistryConfig:
        registry_path = get_env_or_config(ENV_MODEL_REGISTRY_PATH, "artifacts/model_registry.json")
        production_alias = get_env_or_config(ENV_MODEL_REGISTRY_PRODUCTION_ALIAS, "production")
        staging_alias = get_env_or_config(ENV_MODEL_REGISTRY_STAGING_ALIAS, "staging")
        max_versions_to_keep = int(get_env_or_config(ENV_MODEL_REGISTRY_MAX_VERSIONS_TO_KEEP, "10", transform=int))
        quality_gate_max_rmse_degradation_pct = float(get_env_or_config(
            ENV_MODEL_REGISTRY_QUALITY_GATE_MAX_RMSE_DEGRADATION_PCT, "5.0", transform=float
        ))

        registry_config = self.config.model_registry
        use_mlflow = str(get_env_or_config(
            ENV_MLFLOW_USE_MLFLOW,
            str(registry_config.get("use_mlflow", False)),
        )).lower() in ("1", "true", "yes")
        mlflow_tracking_uri = get_env_or_config(
            ENV_MLFLOW_TRACKING_URI,
            registry_config.get("mlflow_tracking_uri", "./mlruns"),
        )
        mlflow_experiment_name = get_env_or_config(
            ENV_MLFLOW_EXPERIMENT_NAME,
            registry_config.get("mlflow_experiment_name", "wine_quality_prediction"),
        )
        mlflow_registry_uri = get_env_or_config(
            ENV_MLFLOW_REGISTRY_URI,
            registry_config.get("mlflow_registry_uri", ""),
        )
        mlflow_model_name = get_env_or_config(
            ENV_MLFLOW_MODEL_NAME,
            registry_config.get("mlflow_model_name", "WineQualityElasticNet"),
        )

        return ModelRegistryConfig(
            registry_path=Path(registry_path),
            production_alias=production_alias,
            staging_alias=staging_alias,
            max_versions_to_keep=max_versions_to_keep,
            quality_gate_max_rmse_degradation_pct=quality_gate_max_rmse_degradation_pct,
            use_mlflow=use_mlflow,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment_name=mlflow_experiment_name,
            mlflow_registry_uri=mlflow_registry_uri,
            mlflow_model_name=mlflow_model_name,
        )
