from src.bone_classifier.config.configuration import configurationManager
from src.bone_classifier.components.model_evaluation import Evaluation
from src.bone_classifier import log 


stage_name="model_evaluation"

class EvaluationPipeline: 
    def __init__(self) -> None:
        pass

    def main(self):
        try: 
            config=configurationManager()
            evaluation_config=config.get_evaluation_config()
            evaluate=Evaluation(evaluation_config)
            evaluate.train_valid_generator()
            evaluate.evaluation()
            evaluate.log_into_mlflow()
        except Exception as e: 
            raise e
        
if __name__ == "__main__":
    log.info(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>{stage_name} started <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    obj=EvaluationPipeline()
    obj.main()
    log.info(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> {stage_name} completed <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    