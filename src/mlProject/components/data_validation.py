import os
from pathlib import Path
import pandas as pd

class DataValidationError(Exception):
    pass

class ValidationResult:
    def __init__(self, schema_valid: bool = True, errors: list = None, drift_detected: bool = False, drift_scores: dict = None):
        self.schema_valid = schema_valid
        self.errors = errors or []
        self.drift_detected = drift_detected
        self.drift_scores = drift_scores or {}

class DataValidation:
    def __init__(self, config):
        self.config = config

    def validate_columns(self, data: pd.DataFrame) -> bool:
        try:
            expected_cols = self.config.get("expected_columns", [])
            missing = [col for col in expected_cols if col not in data.columns]
            if missing:
                raise DataValidationError(f"Missing critical columns: {missing}")
            return True
        except Exception as e:
            print(f"Data validation failed: {e}")
            raise

    def run(self) -> ValidationResult:
        data_file = getattr(self.config, "data_file", Path("artifacts/data_ingestion/winequality-red.csv"))
        if not os.path.exists(data_file):
            return ValidationResult(schema_valid=False, errors=[f"Data file not found: {data_file}"])
        data = pd.read_csv(data_file)
        errors = []
        try:
            self.validate_columns(data)
            schema_valid = True
        except DataValidationError as e:
            schema_valid = False
            errors.append(str(e))
        status_file = getattr(self.config, "STATUS_FILE", Path("artifacts/data_validation/status.txt"))
        os.makedirs(os.path.dirname(status_file), exist_ok=True)
        with open(status_file, "w") as f:
            f.write(f"Validation status: {schema_valid}\n")
            if errors:
                f.write(f"Errors: {', '.join(errors)}\n")
        return ValidationResult(schema_valid=schema_valid, errors=errors)