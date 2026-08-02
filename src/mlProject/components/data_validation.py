import os
import pandas as pd
from dataclasses import dataclass
from mlProject import logger
from mlProject.entity.config_entity import DataValidationConfig


class DataValidationError(Exception):
    pass


@dataclass
class ValidationResult:
    schema_valid: bool
    drift_detected: bool
    errors: list
    drift_scores: dict


class DataValidator:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_columns(self, data: pd.DataFrame) -> bool:
        expected_cols = list(self.config.all_schema.keys())
        missing = [col for col in expected_cols if col not in data.columns]
        if missing:
            raise DataValidationError(f"Missing critical columns: {missing}")
        extra = [col for col in data.columns if col not in expected_cols]
        if extra:
            logger.warning(f"Unexpected columns found: {extra}")
        return True

    def _detect_drift(self, data: pd.DataFrame) -> dict:
        drift_scores = {}
        ref_path = self.config.reference_data_path
        if not os.path.exists(ref_path):
            logger.warning(f"Reference data not found at {ref_path}, skipping drift detection")
            return drift_scores
        ref_data = pd.read_csv(ref_path)
        common_cols = [c for c in data.columns if c in ref_data.columns and c in self.config.all_schema]
        for col in common_cols:
            if data[col].dtype.kind in ('i', 'f') and ref_data[col].dtype.kind in ('i', 'f'):
                ref_mean = ref_data[col].mean()
                cur_mean = data[col].mean()
                ref_std = ref_data[col].std()
                if ref_std > 0:
                    drift_score = abs(cur_mean - ref_mean) / ref_std
                    drift_scores[col] = round(drift_score, 4)
        return drift_scores

    def run(self) -> ValidationResult:
        errors = []
        drift_scores = {}
        schema_valid = False
        drift_detected = False

        try:
            data = pd.read_csv(self.config.data_file)
        except Exception as e:
            errors.append(f"Cannot read data file {self.config.data_file}: {e}")
            return ValidationResult(
                schema_valid=False,
                drift_detected=False,
                errors=errors,
                drift_scores={}
            )

        try:
            self.validate_columns(data)
            schema_valid = True
        except DataValidationError as e:
            errors.append(str(e))
            return ValidationResult(
                schema_valid=False,
                drift_detected=False,
                errors=errors,
                drift_scores={}
            )
        except Exception as e:
            errors.append(f"Unexpected validation error: {e}")
            return ValidationResult(
                schema_valid=False,
                drift_detected=False,
                errors=errors,
                drift_scores={}
            )

        drift_scores = self._detect_drift(data)
        threshold = self.config.drift_threshold
        if drift_scores:
            drift_detected = any(v > threshold for v in drift_scores.values())

        status_file = self.config.STATUS_FILE
        try:
            with open(status_file, "w") as f:
                f.write(f"schema_valid={schema_valid}, drift_detected={drift_detected}")
        except Exception as e:
            logger.warning(f"Could not write status file {status_file}: {e}")

        return ValidationResult(
            schema_valid=schema_valid,
            drift_detected=drift_detected,
            errors=errors,
            drift_scores=drift_scores
        )