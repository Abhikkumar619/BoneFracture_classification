from src.bone_classifier.entity import DataIngestionConfig
from src.bone_classifier import *
import gdown
import zipfile



class DataIngestion: 
    def __init__(self, config: DataIngestionConfig):
        self.config=config
    def download_data(self):
        try: 
            url=self.config.data_url
            url_code=url.split('/')[-2]
            data_path=self.config.root_dir
            os.makedirs(data_path, exist_ok=True)
            perfix_to_download='https://drive.google.com/uc?/export=download&id='+url_code
            log.info(f"Download from url {url} save to path {data_path}")
            gdown.download(perfix_to_download, self.config.data_zip_path)
            log.info(f"Download sucessfully")

        except Exception as e: 
            raise e
        
    def unzip_dir(self):
        try: 
            unzip_path=self.config.unzip_dir
            with zipfile.ZipFile(self.config.data_zip_path, 'r') as zip:
                zip.extractall(unzip_path)
        except Exception as e:
            raise e
