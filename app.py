from flask import Flask, request, jsonify, render_template
import os
from src.bone_classifier.pipeline.prediction import predictionpipeline
from src.bone_classifier.utils.common import decodeImage, encoderImageIntoBase64

app=Flask(__name__)

class Clienapp: 
    def __init__(self) -> None:
        self.file_name="image.jpg"
        self.classifer=predictionpipeline(self.file_name)


@app.route("/", methods=['GET'])
def home(): 
    return render_template('index.html')

@app.route("/train", methods=['GET', 'POST'])
def training(): 
    os.system("python main.py")
    return "Training done sucessfully"


@app.route("/predict", methods=['GET', 'POST'])
def predictRoute(): 
    try:
        image=request.json['image']
        decodeImage(image, clapp.file_name)
        result=clapp.classifer.predict()
        return  jsonify(result)
    except Exception as e:
        raise e




if __name__ == "__main__": 
    clapp=Clienapp()
    app.run(host="0.0.0.0", debug=True, port=5001)
