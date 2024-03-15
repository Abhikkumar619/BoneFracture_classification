from keras.applications.inception_v3 import InceptionV3
import keras
from src.bone_classifier.entity import PrepareBaseModelConfig
from src.bone_classifier import log
from pathlib import Path

class PreparBaseModel: 
    def __init__(self, config: PrepareBaseModelConfig) -> None:
        self.config=config

    @staticmethod
    def save_model(path: Path, model: keras.Model):
        model.save(path)
        log.info(f"Base model save sucessfully at path: {path}")
        

    def get_base_model(self):
        self.model=InceptionV3(
            include_top=self.config.params_include_top,
            weights=self.config.params_weight,
            input_shape=self.config.params_image_size,
            classes=self.config.params_classes)
        
        self.save_model(path=self.config.base_model_path, model=self.model)

    
    def prepare_full_model(self,freez_all, model, freez_till: int ,classes):

        if freez_all:
            for layer in model.layers: 
                model.trainable=False
            log.info(f"Model freez all")

        elif (freez_till is None) and (freez_till>0):
            for layer in model.layers[:-freez_till]:
                model.trainable=False
            log.info(f"Model freetill {freez_till}")


        flatten_in=keras.layers.Flatten()(model.output)
        x=keras.layers.Dense(1000, activation='relu')(flatten_in)
        x=keras.layers.Dropout(0.2)(x)
        x=keras.layers.BatchNormalization()(x)
        x=keras.layers.Dense(500, activation='relu')(x)
        x=keras.layers.Dense(300, activation='relu')(x)
        x=keras.layers.BatchNormalization()(x)
        prediction=keras.layers.Dense(units=5, activation='softmax')(x)

        full_model=keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        # full_model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01), loss=keras.losses.CategoricalCrossentropy, metrics=['acc'])
    

        full_model.summary()

        

        return full_model
    
    def update_base_model(self):
        self.full_model=self.prepare_full_model(freez_all=True, 
                                                model=self.model, 
                                                freez_till=None,
                                                classes=self.config.params_classes)
        
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)





        

