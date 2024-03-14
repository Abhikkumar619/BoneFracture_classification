
from src.bone_classifier.constants import *
from src.bone_classifier.utils.common import read_yaml, create_directories
from src.bone_classifier.entity import DataIngestionConfig


class configurationManager:
    def __init__(self,
                 params_file_path=PARAMS_FILE_PATH,
                 config_file_path= CONFIG_FILE_PATH):
        self.config=read_yaml(config_file_path)
        self.params=read_yaml(params_file_path)

        create_directories([self.config.artifacts_root])

    def get_config_file_path(self)->DataIngestionConfig:

        config=self.config.data_ingestion

        config_file_path=DataIngestionConfig(
            root_dir=config.root_dir,
            data_url=config.data_url,
            data_zip_path=config.data_zip_path,
            unzip_dir=config.unzip_dir)
        return config_file_path
        
        

        