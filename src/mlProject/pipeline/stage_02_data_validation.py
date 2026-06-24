from mlProject.components.data_validation import DataValidation
from mlProject.config.configuration import ConfigurationManager
from mlProject import logger


STAGE_NAME = "Data Validation Stage"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        result = data_validation.run()
        logger.info(f"Validation result: schema_valid={result.schema_valid}, drift_detected={result.drift_detected}")
        if not result.schema_valid:
            raise ValueError(f"Data validation failed due to invalid schema. Errors: {result.errors}")
        if result.drift_detected:
            raise ValueError(f"Data validation failed due to data drift. Scores: {result.drift_scores}")


if __name__ == "__main__":
    pipeline = DataValidationTrainingPipeline()
    pipeline.main()

        