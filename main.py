from src.bone_classifier.pipeline.stage_01_dataingestion import DataIngestionPipeline
from src.bone_classifier import log
from src.bone_classifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelPipeline

stage_name="DataIngestion"
try: 
    log.info(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>{stage_name} started >>>>>>>>>>>>>>>>>>>>>>")
    data_ingestion=DataIngestionPipeline()
    data_ingestion.main()
    log.info(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>{stage_name} completed >>>>>>>>>>>>>>>>>>>>>")
except Exception as e:
    raise e


stage_name = "Prepare_Base_Model"

if __name__ == '__main__': 
    log.info(f">>>>>>>>>>>>>>>>>>>>>> {stage_name} started<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    obj=PrepareBaseModelPipeline()
    obj.main()
    log.info(f">>>>>>>>>>>>>>>>>>>>>>>>>> {stage_name} completed <<<<<<<<<<<<<<<<<<<<")

    


