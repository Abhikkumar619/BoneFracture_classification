import logging
from datetime import datetime
import os
import sys

format_str = "[%(asctime)s : %(levelname)s : %(module)s : %(message)s]"

file_name=f"{datetime.now().strftime('%m_%d_%Y_%H_%S')}.log"

log_file_path=os.path.join(os.getcwd(), "LOG", file_name)

os.makedirs(log_file_path, exist_ok=True)

file_log_path=os.path.join(log_file_path, file_name)


logging.basicConfig(level=logging.INFO, format=format_str,
                    handlers=[logging.FileHandler(file_log_path),
                              logging.StreamHandler(sys.stdout)
                              ])
log=logging.getLogger("bone_fracture")