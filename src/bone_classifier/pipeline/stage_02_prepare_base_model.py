from src.bone_classifier.config.configuration import configurationManager
from src.bone_classifier.components.prepare_base_model import PreparBaseModel
from src.bone_classifier import log 


stage_name="Prepare_Base_Model"

class PrepareBaseModelPipeline: 
    def __init__(self) -> None:
        pass
    
    
    def main(self):
        try: 
            config=configurationManager()
            preparemodel_config=config.get_prepare_base_model()
            prepare_model=PreparBaseModel(preparemodel_config)
            prepare_model.get_base_model()
            prepare_model.update_base_model()
        except Exception as e: 
            raise e
        

if __name__ == '__main__': 
    log.info(f">>>>>>>>>>>>>>>>>>>>>> {stage_name} started<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    obj=PrepareBaseModelPipeline()
    obj.main()
    log.info(f">>>>>>>>>>>>>>>>>>>>>>>>>> {stage_name} completed <<<<<<<<<<<<<<<<<<<<")

    