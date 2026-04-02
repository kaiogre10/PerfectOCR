# main.py
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

TEST_MODE = True

DEFAULT_CONFIG_FILE = os.path.join(PROJECT_ROOT, "config", "master_config.yaml")

LATITUDE_PATH = "D:/outputs/perfectocr"

env_local = "remote" if os.environ.get("CODESPACES") == "true" else "latitude"
if env_local == "remote":
    print("Ejecución remota")
    log_file_paths = os.path.join(PROJECT_ROOT, "perfectocr.txt")
    output_paths = ["output"]
    default_output = output_paths
    
elif os.path.exists(LATITUDE_PATH):
    print("Ejecución en LATITUDE")
    output_paths =[LATITUDE_PATH]
    log_file_paths = "D:/outputs/logs/perfectocr.txt"
    default_output = output_paths

else:
    print("Ejecución en Inspiron")
    output_paths = []
    log_file_paths = ""
    default_output = ["output"]
    
LOG_FILE_PATH = log_file_paths
DEFAULT_OUTPUT_PATH = default_output
OUTPUT_PATH = output_paths

DEFAULT_INPUT_PATH = [
    # "input",
        "input2",
        # "input3"
#  "C:/Users/USER/Desktop/tickets_nuevo"
]

CONSOLE_LEVEL = "INFO"
FILE_LEVEL = "CRITICAL"
CONSOLE_FORMAT = "%(filename)s:%(lineno)d - %(message)s"
FILE_FORMAT = "%(module)s:%(lineno)d - %(message)s"
DATE_FORMAT = "%H:%M" #"%Y-%m-%d %H:%M:%S"

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

if os.path.exists(LOG_FILE_PATH):
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
        output_paths = [os.path.join(PROJECT_ROOT, folder) for folder in OUTPUT_PATH]
        config_path = DEFAULT_CONFIG_FILE
        project_root = PROJECT_ROOT
        
        default_output_paths = [os.path.join(PROJECT_ROOT, folder) for folder in DEFAULT_OUTPUT_PATH]
        clear_output_folders(default_output_paths)
        return activate_main(input_paths, output_paths, config_path, project_root, TEST_MODE)

    return activate_main([], [], "", "", False)

if __name__ == "__main__":
    main()
