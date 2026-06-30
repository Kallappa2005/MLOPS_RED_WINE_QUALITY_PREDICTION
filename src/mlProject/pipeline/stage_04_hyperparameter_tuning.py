from mlProject.config.configuration import ConfigurationManager
from mlProject.components.hyperparameter_tuning import HyperparameterTuner
from mlProject import logger

STAGE_NAME = "Hyperparameter Tuning Stage"

class HyperparameterTuningPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        tuning_config = config.get_hyperparameter_tuning_config()
        tuner = HyperparameterTuner(config=tuning_config)
        tuner.tune()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = HyperparameterTuningPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
