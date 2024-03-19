from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
from src.bone_classifier import log 
from keras.models import load_model


class predictionpipeline: 
    def __init__(self, file_name) -> None:
        self.file_name=file_name
    
    def predict(self): 
        model_path=os.path.join('artifacts/training', 'best_model.h5')
        model=load_model(model_path)
        log.info(f"model loaded sucessfully: {model}")
        test_image=image.load_img(self.file_name, target_size=(224,224))
        test_image=image.img_to_array(test_image)
        test_image=np.expand_dims(test_image, axis=0)
        log.info(f"Test image: {test_image}")


        result=np.argmax(model.predict(test_image), axis=1)
        print(result)

        if result[0]==0: 
            bone="Avulsion fracture"
            return [{"image": bone}]
        elif result[0]==1: 
            bone="Comminuted fracture"
            return [{"image": bone}]
        elif result[0]==2: 
            bone="Fracture Dislocation"
            return [{"image": bone}]
        elif result[0]==3: 
            bone="Greenstick fracture"
            return [{"image": bone}]
        elif result[0]==4: 
            bone="Hairline Fracture"
            return [{"image": bone}]



