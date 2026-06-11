import os
import sys
import logging

CONSOLE_LEVEL = "INFO"
FILE_LEVEL = "INFO"
CONSOLE_FORMAT = "%(asctime)s - %(filename)s:%(lineno)d - %(message)s"
FILE_FORMAT = "%(asctime)s - %(filename)s:%(lineno)d - %(message)s"
DATE_FORMAT = "%H:%M:%S"

# { "filename": "warnings.txt", "level": "WARNING" }, etc.
EXTRA_FILE_LOGS = [
    # {"filename": "errors.txt", "level": "ERROR"},
]

def setup_logging(project_root: str):
    log_root = logging.getLogger()
    log_root.setLevel(logging.DEBUG)

    if log_root.hasHandlers():
        log_root.handlers.clear()

    file_formatter = logging.Formatter(fmt=FILE_FORMAT, datefmt=DATE_FORMAT)
    console_formatter = logging.Formatter(fmt=CONSOLE_FORMAT, datefmt=DATE_FORMAT)

    # Handler principal
    _add_file_handler(log_root, project_root, "perfectocr.txt", FILE_LEVEL, file_formatter)

    # Handlers adicionales por nivel
    for entry in EXTRA_FILE_LOGS:
        _add_file_handler(log_root, project_root, entry["filename"], entry["level"], file_formatter)

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