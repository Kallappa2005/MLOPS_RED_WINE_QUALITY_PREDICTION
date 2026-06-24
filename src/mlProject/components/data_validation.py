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