# services/logs_service.py
import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime
import inspect
from core.assets.patterns import paddle_silence
from typing import Tuple, Optional, Dict, Any
from domain.class_models import TypeModels

_paddle_silence = paddle_silence

LOG_TRACE_FORMAT = "%H:%M:%S"
EXTRA_FILE_LOGS = []
project_root: str = ""

def set_project_root(PROJECT_ROOT: str):
    global project_root
    project_root = PROJECT_ROOT

def _now():
    return datetime.now()

def get_time_stamp(moment: Any, date_format: str):
    return f"{moment.strftime(date_format)}"

def get_caller_info() -> Tuple[str, str]:
    """Nombre[0], Linea_llamadora[1]"""
    frame = inspect.stack()[2]
    return os.path.basename(frame[1]), str(frame[2])

def get_logging_info(get_caller_inf: Tuple[str, str], date_format: str) -> str:
    return f"{get_time_stamp(_now(), date_format)} {get_caller_inf[0]}:{get_caller_inf[1]} -"

def setup_logging(config: Dict[str, str]) -> None:
    """
    Inicializa los descriptores de archivo y la salida por consola.
    Debe invocarse antes de cualquier operación de registro en el sistema.
    """
    console_level = config.get("console_level", "")
    file_level = config.get("file_level", "")
    console_format = config.get("console_format", "")
    file_format = config.get("file_format", "")
    DATE_FORMAT = config.get("date_format", "")
    temp_path_file = config.get("temp_path_file", "")
    
    _log_root = logging.getLogger()
    if _log_root.hasHandlers():
        _log_root.handlers.clear()

    file_formatter = logging.Formatter(fmt=file_format, datefmt=DATE_FORMAT)
    console_formatter = logging.Formatter(fmt=console_format, datefmt=DATE_FORMAT)

    # 1. Configurar Handler Principal en Disco
    _add_file_handler(_log_root, project_root, "perfectocr.txt", file_level, file_formatter)

    # 2. Handlers adicionales si aplican
    if EXTRA_FILE_LOGS:
        for entry in EXTRA_FILE_LOGS: # type: ignore
            _add_file_handler(_log_root, project_root, entry["filename"], entry["level"], file_formatter)

    TEMP_FILE = os.path.join(project_root, temp_path_file, "tmp_file.txt")
    reset_temp_file(TEMP_FILE)

    # 3. Configurar Salida por Consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(console_level.upper())
    _log_root.addHandler(console_handler)

    # 4. Establecer el nivel global definitivo
    _log_root.setLevel(file_level)

    _log_root.propagate = True

def _add_file_handler(log_root: logging.Logger, project_root: str, filename: str, level: str, formatter: logging.Formatter) -> None:
    path = os.path.join(project_root, filename)
    if not os.path.exists(path):
        open(path, "a", encoding=TypeModels.UTF8.value).close()

    handler = logging.FileHandler(path, mode="w", encoding=TypeModels.UTF8.value)
    handler.setFormatter(formatter)
    handler.setLevel(level.upper())
    log_root.addHandler(handler)

def basic_exc_logger(message: str, exc_info: Optional[bool] = False) -> None:
    caller_info = get_caller_info()
    message = f"{get_logging_info(caller_info, LOG_TRACE_FORMAT)} {message}"
    if exc_info:
        print(f"{message}\n{sys.exc_info()}")
    else:
        print(f"{message}")

# def log_temp_process(project_base: Optional[str] = None) -> logging.Logger:
#     caller_info = get_caller_info()[0]
#     if project_base is None or not os.path.isdir(project_base):
#         if not project_root:
#             raise TypeError(f"NO SE INICIALIZARON NI PROJECT ROOT: '{project_root}, NI PROJECT_BASE: '{project_base}'")
        
#         project_base = project_root
        
#     logger = logging.getLogger(caller_info)

#     for handler in logger.handlers[:]:
#         if getattr(handler, "_temp_process_handler", False):
#             logger.removeHandler(handler)
#             handler.close()

#     formatter = logging.Formatter(fmt="%(asctime)s:%(msecs)d %(filename)s:%(lineno)d: %(message)s", datefmt=LOG_TRACE_FORMAT)

#     ext_indx = caller_info.rindex(".")
#     log_path = os.path.join(project_base, "safe_temp", f"{caller_info[:ext_indx]}_temp_log.txt")

#     file_handler = logging.FileHandler(log_path, mode="w", encoding=TypeModels.UTF8.value)
#     file_handler.setLevel(logging.DEBUG)
#     file_handler.setFormatter(formatter)
#     file_handler._temp_process_handler = False  # type: ignore[attr-defined]

#     console_handler = logging.StreamHandler(sys.stdout)
#     console_handler.setLevel(logging.WARNING)
#     console_handler.setFormatter(formatter)
#     console_handler._temp_process_handler = False  # type: ignore[attr-defined]

#     logger.setLevel(logging.DEBUG)
#     logger.propagate = False
#     logger.addHandler(file_handler)
#     logger.addHandler(console_handler)

#     return logger
    
@contextmanager
def intercept_paddle_logs():
    original_write = sys.stderr.write
    try:
        sys.stderr.write = lambda text: original_write(_paddle_silence.sub("", text))
        yield
    finally:
        sys.stderr.write = original_write

def log_simple(msg: str):
    log_info = get_logging_info(get_caller_info(), LOG_TRACE_FORMAT)
    print(f"{log_info} {msg}")

# def format_elapsed_time(seconds: float) -> str:
#     """Convierte segundos a formato HH:MM:SS.ms"""
#     if seconds < 60.0:
#         return f"{seconds:.6f}'s"
#     minutes = int((seconds % 3600) // 60)
#     if minutes < 60:
#         return f"{minutes:02d}:M {seconds % 60:06.3f}'s"
#     else:
#         return f"{int(seconds // 3600):02d}:H {minutes:02d}:M {seconds % 60:06.3f}'s"

def reset_temp_file(TEMP_FILE: str):
    with open(TEMP_FILE, "w", encoding=TypeModels.ASCII.value):
        pass

