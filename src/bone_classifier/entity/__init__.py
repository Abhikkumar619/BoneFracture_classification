from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    data_url: Path
    data_zip_path: Path
    unzip_dir: Path

@dataclass(frozen=True)
class PrepareBaseModelConfig: 
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path

    params_image_size: list
    params_include_top: bool
    params_weight : str
    params_classes: int
    params_learning_rate: int
    
