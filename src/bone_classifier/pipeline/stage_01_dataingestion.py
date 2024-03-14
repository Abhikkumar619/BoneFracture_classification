from src.bone_classifier.config.configuration import configurationManager
from src.bone_classifier.components.data_ingestion import DataIngestion
from src.bone_classifier import *

stage_name= "DataIngestion"

class DataIngestionPipeline:
    def __init__(self) -> None:
        pass
    def main(self):
        try: 
            # Pipline
            configmanager=configurationManager()
            config_file_path=configmanager.get_config_file_path()
            dataingetion=DataIngestion(config_file_path)
            dataingetion.download_data()
            dataingetion.unzip_dir()
        except Exception as e:
            raise e
        
if __name__ == '__main__':
    log.info(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>{stage_name} started >>>>>>>>>>>>>>>>>>>>>>")
    data_ingestion=DataIngestionPipeline()
    data_ingestion.main()
    log.info(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>{stage_name} completed >>>>>>>>>>>>>>>>>>>>>")
