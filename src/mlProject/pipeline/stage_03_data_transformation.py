from pathlib import Path
from mlProject.config.configuration import ConfigurationManager
from mlProject.components.data_transformation import DataTransformation
from mlProject import logger

STAGE_NAME = "Data Transformation Stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        status_file = Path("artifacts/data_validation/status.txt")
        if status_file.exists():
            with open(status_file, "r") as f:
                content = f.read()
            if "Validation status: False" in content:
                raise RuntimeError("Data validation failed. Aborting transformation pipeline. Check artifacts/data_validation/status.txt for details.")
        else:
            raise RuntimeError("Validation status file not found. Run Data Validation stage first.")

        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.train_test_spliting()


if __name__ == "__main__":
    pipeline = DataTransformationTrainingPipeline()
    pipeline.main()