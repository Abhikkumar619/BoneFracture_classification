# BoneFracture_classification

![Sample Image](https://github.com/Abhikkumar619/BoneFracture_classification/blob/main/image.jpg)



## Workflow
1. update config.yaml
2. update params.yaml
3. update entity
4. update configmanager
5. update components
6. update the pipeline
7. update main.py
8. update the dav.yaml
8. app.py

# How to run?

- clone repository: https://github.com/Abhikkumar619/BoneFracture_classification

## steps 01- create a conda environment after the cloning repositoy.
- conda create -p env_name python=3.8 -y

## steps 02 - install the requirements
- pip install -r requirement.txt

## Finally run the command 
- python app.py


## Dagshub
MLFLOW_TRACKING_URI=https://dagshub.com/Abhikkumar619/BoneFracture_classification.mlflow \
MLFLOW_TRACKING_USERNAME=Abhikkumar619 \
MLFLOW_TRACKING_PASSWORD=b515b18fe70cac23bd1c8591a7c54e188845b00c \
python script.py

## Run this to export env variable
- export MLFLOW_TRACKING_URI=https://dagshub.com/Abhikkumar619/BoneFracture_classification.mlflow
- export MLFLOW_TRACKING_USERNAME=Abhikkumar619 
- export MLFLOW_TRACKING_PASSWORD=b515b18fe70cac23bd1c8591a7c54e188845b00c


# DVC commmad (data version control)
- dvc init 
- dvc repr
- dvc dag (to see graph of dvc pipeline.)

