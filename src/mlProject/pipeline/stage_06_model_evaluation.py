from mlProject.components.model_evaluation import ModelEvaluation
from mlProject.config.configuration import ConfigurationManager



STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluator = ModelEvaluation(config=model_evaluation_config)
        model_evaluator.save_results()


if __name__ == "__main__":
    pipeline = ModelEvaluationPipeline()
    pipeline.main()
