# services/logs_service.py
import os
import sys
from datetime import datetime
import logging
from core.utils.patterns import float_time
import inspect
from typing import List, Tuple, Optional

_float_time = float_time
CONSOLE_LEVEL = 'INFO'
FILE_LEVEL = 'INFO'
CONSOLE_FORMAT = "%(asctime)s - %(filename)s:%(lineno)d - %(message)s"
FILE_FORMAT = "%(filename)s:%(lineno)d - %(message)s"
DATE_FORMAT = "%H:%M:%S"
TEMP_DATE_FORMAT = "%Y/%m/%d %H:%M:%S"
TEMP_PATH_FILE: str = "temp"
TEMP_FILE: str
EXTRA_FILE_LOGS = []

def get_time_stamp(log: bool):
    now = datetime.now()
    return f"{now.strftime((DATE_FORMAT if log else TEMP_DATE_FORMAT))}"

def get_caller_info() -> Tuple[str, str]:
    """Devuelve path, linea"""
    frame = inspect.stack()[2]
    return os.path.basename(frame[1]), str(frame[2])

def get_logging_info(get_caller_info: Tuple[str, str]) -> str:
    return f"{get_time_stamp(True)} - {get_caller_info[0]}:{get_caller_info[1]} "

def setup_logging(project_root: str) -> None:
    """
    Inicializa los descriptores de archivo y la salida por consola.
    Debe invocarse antes de cualquier operación de registro en el sistema.
    """
    _log_root = logging.getLogger()
    if _log_root.hasHandlers():
        _log_root.handlers.clear()

    file_formatter = logging.Formatter(fmt=FILE_FORMAT, datefmt=DATE_FORMAT)
    console_formatter = logging.Formatter(fmt=CONSOLE_FORMAT, datefmt=DATE_FORMAT)

    # 1. Configurar Handler Principal en Disco
    _add_file_handler(_log_root, project_root, "perfectocr.txt", FILE_LEVEL, file_formatter)

    # 2. Handlers adicionales si aplican
    for entry in EXTRA_FILE_LOGS: # type: ignore
        _add_file_handler(_log_root, project_root, entry["filename"], entry["level"], file_formatter) # type: ignore

    TEMP_FILE = os.path.join(project_root, TEMP_PATH_FILE, "tmp_file.txt") # type: ignore
    reset_temp_file(TEMP_FILE)

    # 3. Configurar Salida por Consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(CONSOLE_LEVEL.upper())
    _log_root.addHandler(console_handler)

    # 4. Establecer el nivel global definitivo
    _log_root.setLevel('DEBUG')

def _add_file_handler(log_root: logging.Logger, project_root: str, filename: str, level: str, formatter: logging.Formatter) -> None:
    path = os.path.join(project_root, filename)
    if not os.path.exists(path):
        open(path, "a", encoding="utf-8").close()

    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setFormatter(formatter)
    handler.setLevel(level.upper())
    log_root.addHandler(handler)

def log_active_areas(message: str, manager_config: Optional[List[Tuple[str, List[str]]]] = None) -> None:
    caller_info = get_caller_info()
    message = f"{get_logging_info(caller_info)} {message}"
    if manager_config:
        stages_list = [stager[0].removesuffix("_stage").title() for stager in manager_config if stager and stager[1]]
        msg = f"{', '.join(stages_list) if stages_list else 'SOLO BUILDERS'}"
        print(f"{message} '{msg}'")
    else:
        print(f"{message}")

def basic_exc_logger(message: str, exc_info: Optional[bool] = False) -> None:
    caller_info = get_caller_info()
    message = f"{get_logging_info(caller_info)} {message}"
    if exc_info:
        print(f"{message}\n{sys.exc_info()}")
    else:
        print(f"{message}")
    
# def intercept_paddle_logs():
#     sys.stderr.write = lambda text, orig=sys.stderr.write: orig(_paddle_silene.sub("", text))

def log_simple(msg: str):
    log_info = get_logging_info(get_caller_info())
    print(f"{log_info} {msg}")

class TiempoFilter(logging.Filter):
    def filter(self, record: logging.LogRecord):
        if (record.levelno == logging.WARNING and record.module == "main_builder"):
            msg = record.getMessage()
            matches = _float_time.finditer(msg)
            if matches:
                text_float = float([(match.group()[7:]) for match in matches if match.group()][0].strip())
                record.msg = f"Total: {format_elapsed_time(text_float)}"
                record.args = ()

        return True

def format_elapsed_time(seconds: float) -> str:
    """Convierte segundos a formato HH:MM:SS.ms"""
    if seconds < 60.0:
        return f"{seconds:.6f}'s"
    minutes = int((seconds % 3600) // 60)
    if minutes < 60:
        return f"{minutes:02d}:M {seconds % 60:06.3f}'s"
    else:
        return f"{int(seconds // 3600):02d}:H {minutes:02d}:M {seconds % 60:06.3f}'s"

# class TimeWorker(logging.Filter):
#     def filter(self, record: logging.LogRecord):
#
#         if (record.levelno == logging.WARNING and record.module == "main_builder"):
#             msg = record.getMessage()
#             matches = _float_time.finditer(msg)
#             if matches:
#                 text_float = float([(match.group()[7:]) for match in matches if match.group()][0].strip())
#                 record.msg = f"Total: {format_elapsed_time(text_float)}"
#                 record.args = ()
#
#         return True

def reset_temp_file(TEMP_FILE: str):
    with open(TEMP_FILE, "w", encoding="utf-8"):
        pass
