from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    expected_checksum: str = ""
    override_root_dir: Optional[Path] = None
    override_source_URL: Optional[str] = None

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: Path
    data_file: Path
    all_schema: dict
    drift_threshold: float = 0.05
    reference_data_path: Path = Path("artifacts/reference_data.csv")
    override_root_dir: Optional[Path] = None
    override_drift_threshold: Optional[float] = None

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    test_size: float
    random_state: int
    stratify_column: Optional[str]
    use_scaler: bool = True
    scaler_type: str = "standard"
    handle_outliers: bool = True
    outlier_method: str = "iqr"
    outlier_iqr_multiplier: float = 1.5
    impute_missing: bool = True
    feature_engineering_flags: Optional[dict] = None
    preprocessor_path: Optional[Path] = None
    min_samples_per_class: int = 4
    enable_per_class_evaluation: bool = True
    override_root_dir: Optional[Path] = None
    override_test_size: Optional[float] = None

@dataclass(frozen=True)
class HyperparameterTuningConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    target_column: str
    preprocessor_path: Optional[Path] = None
    use_scaler: bool = True
    n_trials: int = 20
    alpha_min: float = 0.001
    alpha_max: float = 2.0
    l1_ratio_min: float = 0.0
    l1_ratio_max: float = 1.0

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    alpha: float
    l1_ratio: float
    target_column: str
    preprocessor_path: Optional[Path] = None
    use_scaler: bool = True
    override_root_dir: Optional[Path] = None
    override_alpha: Optional[float] = None
    override_l1_ratio: Optional[float] = None

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str
    preprocessor_path: Optional[Path] = None
    use_scaler: bool = True
    per_class_r2_threshold: float = -0.5
    override_root_dir: Optional[Path] = None

@dataclass(frozen=True)
class ModelRegistryConfig:
    registry_path: Path
    production_alias: str = "production"
    staging_alias: str = "staging"
    max_versions_to_keep: int = 10
    quality_gate_max_rmse_degradation_pct: float = 5.0
    use_mlflow: bool = False
    mlflow_tracking_uri: str = "./mlruns"
    mlflow_experiment_name: str = "wine_quality_prediction"
    mlflow_registry_uri: str = ""
    mlflow_model_name: str = "WineQualityElasticNet"
