import os
from pathlib import Path
import pandas as pd

class DataValidationError(Exception):
    pass

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


def write_validation_status(valid: bool, status_path: str = "artifacts/data_validation/status.txt", errors: list = None):
    os.makedirs(os.path.dirname(status_path), exist_ok=True)
    with open(status_path, "w") as f:
        f.write(f"Validation status: {valid}\n")
        if errors:
            f.write(f"Errors: {', '.join(errors)}\n")