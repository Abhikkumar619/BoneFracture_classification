from src.bone_classifier.config.configuration import configurationManager
from src.bone_classifier.components.model_training import ModelTraining
from src.bone_classifier import log


stage_name="Model_training"

class ModelTrainingPipeline:
    def __init__(self) -> None:
        pass
    def main(self):
        try: 
            config=configurationManager()
            model_training_config=config.get_model_training_config()
            model_training=ModelTraining(model_training_config)
            model_training.get_base_model()
            model_training.train_valid_generator()
            model_training.train()
        except Exception as e: 
            raise e 
        
if __name__ == "__main__":
    log.info(f">>>...........................{stage_name} started................................")
    obj=ModelTrainingPipeline()
    obj.main()
    log.info(f">>>............................{stage_name} completed >>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    