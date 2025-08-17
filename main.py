# PerfectOCR/main.py
import time
START_TIME = time.perf_counter()

# Log cada import individualmente
IMPORT_START = time.perf_counter()
import logging
LOGGING_IMPORT_END = time.perf_counter()
print(f"Import logging: {LOGGING_IMPORT_END - IMPORT_START:.6f}s")

import os
OS_IMPORT_END = time.perf_counter()
print(f"Import os: {OS_IMPORT_END - LOGGING_IMPORT_END:.6f}s")

import sys
SYS_IMPORT_END = time.perf_counter()
print(f"Import sys: {SYS_IMPORT_END - OS_IMPORT_END:.6f}s")

import typer
TYPER_IMPORT_END = time.perf_counter()
print(f"Import typer: {TYPER_IMPORT_END - SYS_IMPORT_END:.6f}s")

from typing import Optional, List
TYPING_IMPORT_END = time.perf_counter()
print(f"Import typing: {TYPING_IMPORT_END - TYPER_IMPORT_END:.6f}s")

from services.cache_service import clear_output_folders
CACHE_IMPORT_END = time.perf_counter()
print(f"Import cache_service: {CACHE_IMPORT_END - TYPING_IMPORT_END:.6f}s")

from app.main_builder import activate_main
MAIN_BUILDER_IMPORT_END = time.perf_counter()
print(f"Import main_builder: {MAIN_BUILDER_IMPORT_END - CACHE_IMPORT_END:.6f}s")

# Log después de todos los imports
ALL_IMPORTS_END = time.perf_counter()
print(f"TODOS los imports completados en {ALL_IMPORTS_END - START_TIME:.6f}s")

logger = logging.getLogger(__name__)

# Log después de imports
IMPORTS_END = time.perf_counter()
logging.info(f"Imports completados en {IMPORTS_END - START_TIME:.6f}s")

os.environ.update({
    'OMP_NUM_THREADS': '1',        
    'MKL_NUM_THREADS': '2',
    'FLAGS_use_mkldnn': '1',     
})

# Log después de variables de entorno
ENV_TIME = time.perf_counter()
logging.info(f"Variables de entorno configuradas en {ENV_TIME - IMPORTS_END:.6f}s")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Log después de configuración de paths
PATHS_TIME = time.perf_counter()
logging.info(f"Paths configurados en {PATHS_TIME - ENV_TIME:.6f}s")

DEFAULT_CONFIG_FILE = os.path.join(PROJECT_ROOT, "config", "master_config.yaml")
DEFAULT_INPUT_PATH = os.path.join(PROJECT_ROOT, "input")
DEFAULT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output")
LOG_FILE_PATH = os.path.join(PROJECT_ROOT, "perfectocr.txt")

"""Configura el logging global usando parámetros definidos manualmente aquí."""

CONSOLE_LEVEL = "INFO"
FILE_LEVEL = "DEBUG"
# Formato para la consola: muestra el nivel, nombre del logger, línea y mensaje
CONSOLE_FORMAT = "%(filename)s:%(lineno)d - %(message)s"
# Formato para el archivo: incluye fecha/hora, --nivel--, --nombre del logger--, --línea--, --módulo-- y mensaje
FILE_FORMAT = "%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(module)s - %(message)s"
# Formato de fecha/hora utilizado en los logs
DATE_FORMAT = "%Y-%m-%d %H:%M:%S" #"%Y-%m-%d %H:%M:%S"

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

# Log después de configuración de logging
LOGGING_TIME = time.perf_counter()
logging.info(f"Logging configurado en {LOGGING_TIME - PATHS_TIME:.6f}s")

# Log antes de crear la app Typer
logging.info("Creando aplicación Typer...")
TYPER_START = time.perf_counter()
app = typer.Typer(help="PerfectOCR - Sistema de OCR optimizado")
TYPER_TIME = time.perf_counter()
logging.info(f"App Typer creada en {TYPER_TIME - TYPER_START:.6f}s")

# Log final
FINAL_TIME = time.perf_counter()
logging.info(f"Arranque de sistema en {FINAL_TIME - START_TIME:.6f}s")


@app.command()
def run(
    input_paths: Optional[List[str]] = typer.Argument(None),
    output_paths: Optional[List[str]] = typer.Argument(None),
    config_path: str = typer.Option(DEFAULT_CONFIG_FILE, "--config", "-c"),
    project_root: str = PROJECT_ROOT
):
    """Ejecuta PerfectOCR con rutas por defecto o personalizadas."""
    if not input_paths:
        input_paths = [DEFAULT_INPUT_PATH]
    if not output_paths:
        output_paths = [DEFAULT_OUTPUT_PATH]
    
    clear_output_folders(output_paths, project_root)
    return activate_main(input_paths, output_paths, config_path, project_root)

def main():
    """Función main para compatibilidad con ejecución directa."""
    if len(sys.argv) == 1:
        input_paths = [DEFAULT_INPUT_PATH]
        output_paths = [DEFAULT_OUTPUT_PATH]
        config_path = DEFAULT_CONFIG_FILE
        project_root = PROJECT_ROOT
        
        clear_output_folders(output_paths, project_root)
        return activate_main(input_paths, output_paths, config_path, project_root)
    app()

if __name__ == "__main__":
    main()