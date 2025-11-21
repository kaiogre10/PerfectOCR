
# PerfectOCR/main.py
import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import logging
from services.cache_service import clear_output_folders
from app.main_builder import activate_main

logger = logging.getLogger(__name__)

os.environ.update({
    'OMP_NUM_THREADS': '4',
    'MKL_NUM_THREADS': '4',
    'FLAGS_use_mkldnn': '1',
})

DEFAULT_CONFIG_FILE = os.path.join(PROJECT_ROOT, "config", "master_config.yaml")
DEFAULT_INPUT_PATH = [
"input",
  # "input2",
#   "input3",
#  "C:/Users/USER/Desktop/tickets_nuevo"
]

DEFAULT_OUTPUT_PATH =[
    "output"
]

LOG_FILE_PATH = os.path.join(PROJECT_ROOT, "perfectocr.txt")
CONSOLE_LEVEL = "INFO"
FILE_LEVEL = "INFO"
CONSOLE_FORMAT = "%(filename)s:%(lineno)d - %(message)s"
FILE_FORMAT = "%(asctime)s - %(module)s:%(lineno)d - %(message)s"
DATE_FORMAT = "%D - %H:%M" #"%Y-%m-%d %H:%M:%S"

logger_root = logging.getLogger()
logger_root.setLevel(logging.DEBUG)
if logger_root.hasHandlers():
    logger_root.handlers.clear()

file_formatter = logging.Formatter(
    fmt=FILE_FORMAT,
    datefmt=DATE_FORMAT
)
console_formatter = logging.Formatter(
    fmt=CONSOLE_FORMAT
)

file_handler = logging.FileHandler(LOG_FILE_PATH, mode='w', encoding='utf-8')
file_handler.setFormatter(file_formatter)
file_handler.setLevel(FILE_LEVEL.upper())
logger_root.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(console_formatter)
console_handler.setLevel(CONSOLE_LEVEL.upper())
logger_root.addHandler(console_handler)

def main():
    """Función main para compatibilidad con ejecución directa."""
    if len(sys.argv) == 1:
        input_paths = [os.path.join(PROJECT_ROOT, folder) for folder in DEFAULT_INPUT_PATH]
        output_paths = [os.path.join(PROJECT_ROOT, folder) for folder in DEFAULT_OUTPUT_PATH]
        config_path = DEFAULT_CONFIG_FILE
        project_root = PROJECT_ROOT
        
        clear_output_folders(output_paths, project_root)
        return activate_main(input_paths, output_paths, config_path, project_root)

if __name__ == "__main__":
    main()