from src.bone_classifier.pipeline.stage_01_dataingestion import DataIngestionPipeline
from src.bone_classifier import log


stage_name="DataIngestion"
try: 
    log.info(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>{stage_name} started >>>>>>>>>>>>>>>>>>>>>>")
    data_ingestion=DataIngestionPipeline()
    data_ingestion.main()
    log.info(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>{stage_name} completed >>>>>>>>>>>>>>>>>>>>>")
except Exception as e:
    raise e


