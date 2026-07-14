import pandas as pd
from mlProject.entity.config_entity import DataValidationConfig


ValidationResult = namedtuple("ValidationResult", [
    "schema_valid", "drift_detected", "drift_scores", "errors"
])


class SchemaValidator:
    def __init__(self, schema: dict):
        self.schema = schema

    def validate(self, data: pd.DataFrame) -> tuple[bool, list[str]]:
        errors = []
        schema_cols = dict(self.schema)
        for col in data.columns:
            if col not in schema_cols:
                errors.append(f"Unexpected column '{col}' found in data")
                continue
            expected_dtype = schema_cols[col]
            actual_dtype = str(data[col].dtype)
            # Use type-family checks to handle NaN-induced upcasting (e.g. int64 -> float64)
            dtype_ok = (actual_dtype == expected_dtype) or (
                "int" in expected_dtype and pd.api.types.is_numeric_dtype(data[col])
            ) or (
                "float" in expected_dtype and pd.api.types.is_float_dtype(data[col])
            )
            if not dtype_ok:
                errors.append(
                    f"Column '{col}' type mismatch: expected {expected_dtype}, got {actual_dtype}"
                )
            null_count = data[col].isnull().sum()
            if null_count > 0:
                errors.append(f"Column '{col}' has {null_count} null value(s)")
        for col in schema_cols:
            if col not in data.columns:
                errors.append(f"Missing expected column '{col}' in data")
        return len(errors) == 0, errors


class DriftDetector:
    def __init__(self, threshold: float = 0.05, root_dir: Path = None):
        self.threshold = threshold
        self.root_dir = root_dir

    def detect(self, reference: pd.DataFrame, production: pd.DataFrame) -> tuple[bool, dict[str, float]]:
        try:
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset
            
            ref_features = reference.drop(columns=["quality"], errors="ignore")
            prod_features = production.drop(columns=["quality"], errors="ignore")
            
            report = Report(metrics=[
                DataDriftPreset(stattest_threshold=self.threshold)
            ])
            
            report.run(reference_data=ref_features, current_data=prod_features)
            
            if self.root_dir:
                report_path = Path(self.root_dir) / "drift_report.html"
                report.save_html(str(report_path))
                logger.info(f"Evidently AI drift report saved to {report_path}")
                
            report_dict = report.as_dict()
            dataset_drift = report_dict["metrics"][0]["result"]["dataset_drift"]
            
            scores = {}
            drift_by_columns = report_dict["metrics"][0]["result"]["drift_by_columns"]
            for col, metrics in drift_by_columns.items():
                scores[col] = round(metrics.get("drift_score", 0), 6)
                
            return dataset_drift, scores
        except ImportError:
            logger.error("evidently library is not installed. Run: pip install evidently")
            raise
        except Exception as e:
            logger.error(f"Evidently AI drift detection failed: {e}")
            raise

class DataValidationError(Exception):
    pass

class DataValidation:
    def __init__(self, config):
        self.config = config
        self.schema_validator = SchemaValidator(config.all_schema)
        self.drift_detector = DriftDetector(config.drift_threshold, config.root_dir)

    def run(self) -> ValidationResult:
        try:
            expected_cols = self.config.get("expected_columns", [])
            missing = [col for col in expected_cols if col not in data.columns]
            if missing:
                raise DataValidationError(f"Missing critical columns: {missing}")
            return True
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            raise