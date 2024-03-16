import keras
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from src.bone_classifier.entity import ModelTrainingConfig



class ModelTraining: 
    def __init__(self, config: ModelTrainingConfig) -> None:
        self.config=config
    

    def get_base_model(self): 
        self.model=keras.models.load_model(self.config.trained_model_path)
    

    def train_valid_generator(self):

        train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)
        validation_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)

        self.train_generator= train_datagen.flow_from_directory(
            self.config.train_data_path,
            target_size=(224,224),
            batch_size=self.config.batch_size
            
        )
        
        self.valid_generator=validation_datagen.flow_from_directory(
            self.config.valid_data_path,
            target_size=(224,224),
            batch_size=self.config.batch_size
        )
    @staticmethod
    def save_model(path: Path, model: keras.models):
        model.save(path)

    

    def train(self):


        self.step_per_epoch=self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps=self.valid_generator.samples // self.valid_generator.batch_size


        
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model.fit(self.train_generator, 

                       epochs=self.config.epoch,
                       steps_per_epoch=self.step_per_epoch,
                       validation_data=self.valid_generator,
                       validation_steps=self.validation_steps)
        
        self.save_model(path=self.config.final_model_path,
                         model=self.model)

    