from collections import namedtuple
from scipy.stats import ks_2samp
from pathlib import Path

from mlProject import logger
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
    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold

    def detect(self, reference: pd.DataFrame, production: pd.DataFrame) -> tuple[bool, dict[str, float]]:
        scores = {}
        drift = False
        common_cols = [c for c in reference.columns if c in production.columns and c != "quality"]
        for col in common_cols:
            ref_dtype = reference[col].dtype
            if not pd.api.types.is_numeric_dtype(ref_dtype):
                continue
            ref_clean = reference[col].dropna()
            prod_clean = production[col].dropna()
            if len(ref_clean) == 0 or len(prod_clean) == 0:
                continue
            stat, p_value = ks_2samp(ref_clean, prod_clean)
            scores[col] = round(p_value, 6)
            if p_value < self.threshold:
                drift = True
        return drift, scores


class DataValidator:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        self.schema_validator = SchemaValidator(config.all_schema)
        self.drift_detector = DriftDetector(config.drift_threshold)

    def run(self) -> ValidationResult:
        try:
            data = pd.read_csv(self.config.data_file)
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.config.data_file}")
            raise
        except Exception as e:
            logger.exception(f"Failed to read data file: {self.config.data_file}")
            raise

        schema_valid, schema_errors = self.schema_validator.validate(data)
        
        # Load reference data (training distribution) for drift detection
        reference_data_path = self.config.reference_data_path
        if reference_data_path.exists():
            try:
                reference_data = pd.read_csv(reference_data_path)
                logger.info(f"Loaded reference data from {reference_data_path}")
            except Exception as e:
                logger.warning(f"Failed to load reference data: {e}")
                reference_data = None
        else:
            if schema_valid:
                logger.info(f"Reference data not found at {reference_data_path}. Saving current data as reference for future runs.")
                data.to_csv(reference_data_path, index=False)
                logger.info(f"Created reference data snapshot at {reference_data_path}")
            else:
                logger.warning(f"Skipping reference data save — schema validation failed. Errors: {schema_errors}")
            reference_data = None

        if reference_data is not None:
            drift_detected, drift_scores = self.drift_detector.detect(reference_data, data)
            if drift_detected:
                logger.error(
                    f"Data drift detected! Threshold: {self.config.drift_threshold}. "
                    f"Drift scores: {drift_scores}"
                )
        else:
            drift_detected = False
            drift_scores = {}

        all_errors = list(schema_errors)
        validation_status = schema_valid and not drift_detected

        with open(self.config.STATUS_FILE, 'w') as f:
            f.write(f"Validation status: {validation_status}\n")
            f.write(f"Schema valid: {schema_valid}\n")
            f.write(f"Drift detected: {drift_detected}\n")
            if drift_scores:
                f.write("Drift scores:\n")
                for col, pv in drift_scores.items():
                    f.write(f"  {col}: {pv}\n")
            if all_errors:
                f.write("Errors:\n")
                for err in all_errors:
                    f.write(f"  - {err}\n")

        logger.info(
            f"Validation {'passed' if validation_status else 'failed'}: "
            f"schema_valid={schema_valid}, drift_detected={drift_detected}, "
            f"errors={len(all_errors)}"
        )
        return ValidationResult(schema_valid, drift_detected, drift_scores, all_errors)
