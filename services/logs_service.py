import os
import sys
import logging
import inspect
from datetime import datetime
from typing import List, Tuple, Optional
# from core.utils.text_utils import format_elapsed_time
import re
# # Sobrescribe la escritura de errores para borrar el aviso si aparece

logger = logging.getLogger(__name__)
    
PROJECT_ROOT: str = ""
CONSOLE_LEVEL = 'INFO'
FILE_LEVEL = 'INFO'
CONSOLE_FORMAT = "%(asctime)s - %(filename)s:%(lineno)d - %(message)s"
FILE_FORMAT = "%(asctime)s - %(filename)s:%(lineno)d - %(message)s"
DATE_FORMAT = "%H:%M:%S"

# { "filename": "warnings.txt", "level": "WARNING" }, etc.
EXTRA_FILE_LOGS = [
    # {"filename": "errors.txt", "level": "ERROR"},
]

def set_project_root(project_root: str):
    global PROJECT_ROOT
    PROJECT_ROOT = project_root # type: ignore

def get_time_stamp():
    now = datetime.now()
    return f"{now.strftime(DATE_FORMAT)}"

def get_caller_info() -> Tuple[str, str]:
    frame = inspect.stack()[2]
    return os.path.basename(frame[1]), str(frame[2])

def setup_logging():
    log_root = logging.getLogger()
    log_root.setLevel('INFO')

    if log_root.hasHandlers():
        log_root.handlers.clear()

    file_formatter = logging.Formatter(fmt=FILE_FORMAT, datefmt=DATE_FORMAT)
    console_formatter = logging.Formatter(fmt=CONSOLE_FORMAT, datefmt=DATE_FORMAT)

    # Handler principal
    _add_file_handler(log_root, PROJECT_ROOT, "perfectocr.txt", FILE_LEVEL, file_formatter)

    # Handlers adicionales por nivel
    for entry in EXTRA_FILE_LOGS: # type: ignore
        _add_file_handler(log_root, PROJECT_ROOT, entry["filename"], entry["level"], file_formatter) # type: ignore

    # Consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(CONSOLE_LEVEL.upper())
    log_root.addHandler(console_handler)

def _add_file_handler(log_root: logging.Logger, project_root: str, filename: str, level: str, formatter: logging.Formatter) -> None:
    path = os.path.join(project_root, filename)
    if not os.path.exists(path):
        open(path, "a", encoding="utf-8").close()

    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setFormatter(formatter)
    handler.setLevel(level.upper())
    log_root.addHandler(handler)

def log_simple(message: str) -> None:
    info = get_logging_info(get_caller_info())
    print(f"{info} {message}")

def log_active_areas(message: str, manager_config: Optional[List[Tuple[str, List[str]]]] = None) -> None:
    message = f"{get_logging_info(get_caller_info())} {message}"
    if manager_config:
        stages_list: List[str] = []
        for stager in manager_config:
            stage = stager[0]

            workers = stager[1]
            if not stager or not workers:
                continue
            stage = stage.removesuffix("_stage").title()
            stages_list.append(stage)
        msg = f"{", ".join(stages_list) if stages_list else "SOLO BUILDERS"}"
        print(f"{message}'{msg}'")
    else:
        print(f"{message}")
    
def get_logging_info(get_caller_info: Tuple[str, str]):
    return f"{get_time_stamp()} - {get_caller_info[0]}:{get_caller_info[1]}"

def intercept_paddle_logs():
    sys.stderr.write = lambda text, orig=sys.stderr.write: orig(re.sub(r".*OMP_NUM_THREADS.*|.*PLEASE USE OMP_NUM_THREADS WISELY.*", "", text)) # type: ignore

def basic_logger(message: str):
    print(f"{get_logging_info(get_caller_info())} {message}")