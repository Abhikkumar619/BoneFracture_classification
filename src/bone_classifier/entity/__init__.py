from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    data_url: Path
    data_zip_path: Path
    unzip_dir: Path
