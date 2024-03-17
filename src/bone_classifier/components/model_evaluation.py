import mlflow
import mlflow.keras
from urllib.parse import urlparse
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from pathlib import Path
from src.bone_classifier.entity import EvaluationConfig
import tensorflow as tf
from src.bone_classifier import log 
from src.bone_classifier.utils.common import save_json



class Evaluation: 
    def __init__(self, config: EvaluationConfig) -> None:
        self.config=config
        
    def train_valid_generator(self):
        
        train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
        validation_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)

        self.train_generator= train_datagen.flow_from_directory(
            self.config.train_data_path,
            target_size=(224,224),
            batch_size=self.config.params_batch_size
            
        )
        
        self.valid_generator=validation_datagen.flow_from_directory(
            self.config.valid_data_path,
            target_size=(224,224),
            batch_size=self.config.params_batch_size
        )
    @staticmethod
    def load_models(path: Path): 
        return tf.keras.models.load_model(path)
    
    def save_score(self): 
        score={'loss': self.score[0], "accuracy": self.score[1]}
        log.info(f"Score: {score}")
        save_json(path=Path("scores.json"), data=score)

    def evaluation(self): 
        self.model=self.load_models(self.config.path_of_model)
        self.train_valid_generator()
        self.score=self.model.evaluate(self.valid_generator)
        log.info(f"Evaluation score: {self.score}")
        self.save_score()

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme

        log.info(f"tracking url type store { tracking_url_type_store}")

        with mlflow.start_run():
            mlflow.log_params(self.config.all_parmas)
            mlflow.log_metrics(
                {"loss":  self.score[0], "accuracy": self.score[1]}
            )
            if tracking_url_type_store != 'file':
                mlflow.keras.log_model(self.model, artifact_path="model", registered_model_name="inception_v3")
            else:
                mlflow.keras.log_model(self.model, artifact_path="model")
