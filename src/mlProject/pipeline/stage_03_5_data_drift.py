from mlProject.config.configuration import ConfigurationManager
from mlProject.components.data_drift import DataDrift
from mlProject import logger

STAGE_NAME = "Data Drift Stage"

class DataDriftPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_drift_config = config.get_data_drift_config()
        data_drift = DataDrift(config=data_drift_config)
        data_drift.detect_drift()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataDriftPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
