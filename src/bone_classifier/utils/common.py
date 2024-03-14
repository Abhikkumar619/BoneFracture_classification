import os
from box.exceptions import BoxValueError # Exception handling.
import yaml      # To read yaml file.
import joblib  # To 
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64
import json
from src.bone_classifier import log # For logging.


@ensure_annotations
def read_yaml(path_to_yaml:Path) -> ConfigBox:
    """
    reads yaml file and return 
    """
    try: 
        with open(path_to_yaml) as yaml_file:
            content=yaml.safe_load(yaml_file)
            log.info(f"yaml file : {path_to_yaml} loaded sucessfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e

@ensure_annotations
def create_directories(path_to_directories  : list, verbose=True):
    for file in path_to_directories:
        os.makedirs(file, exist_ok=True)

        if verbose:
            log.info(f"Created directories at : {file}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """
    Save JSON data to a file.
    """
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=4)  # 'indent' is optional and makes the JSON file more readable

    # logger.info(f"Json file saved at {path}")



@ensure_annotations
def load_json(path: Path):
    """
    load the json file.
    """
    with open(path) as f:
        content=json.load(f)
    log.info(f"Json file loaded from path {path} sucessfully")
    
    return ConfigBox(content)


# If we want to save file in binary format. 

@ensure_annotations
def save_bin(data: Any, path: Path):
    """
    Save binary file.
    """
    joblib.dump(value=data, filename=path)
    log.info(f"Binary file saved ar: {path}")


# Load the binary file.
@ensure_annotations
def load_bin(path: Path):
    """
    load Binary file.
    """
    data=joblib.load(path)
    log.info(f"binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str: 
    """
    get size in kb
    """
    size_in_kb=round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"


def decodeImage(imgstring, filename):
    imgdata=base64.b64decode(imgstring)
    with open(filename, 'wb') as f: 
        f.write(imgdata)
        f.close()

def encoderImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f: 
        return base64.b64decode(f.read())