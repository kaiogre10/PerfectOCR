# PerfectOCR/main.py
import logging
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# import typer
# from typing import Optional, List
from services.cache_service import clear_output_folders
from app.main_builder import activate_main

logger = logging.getLogger(__name__)

os.environ.update({
    'OMP_NUM_THREADS': '1',        
    'MKL_NUM_THREADS': '2',
    'FLAGS_use_mkldnn': '1',     
})

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

# app = typer.Typer(help="PerfectOCR - Sistema de OCR optimizado")

"""Hint de uso para la terminal:
Ejemplo de uso:   
python main.py run --input-paths "ruta/imagen1.png" "ruta/imagen2.jpg" --output-paths "ruta/salida1" "ruta/salida2" --config "config/master_config.yaml" --project-root "C:/PerfectOCR"
Todos los argumentos son opcionales; si no se especifican, se usan los valores por defecto definidos en el sistema."""

# @app.command()
# def run(
#     input_paths: Optional[List[str]] = typer.Argument(None),
#     output_paths: Optional[List[str]] = typer.Argument(None),
#     config_path: str = typer.Option(DEFAULT_CONFIG_FILE, "--config", "-c"),
#     project_root: str = PROJECT_ROOT
# ):
#     """Ejecuta PerfectOCR con rutas por defecto o personalizadas."""
#     if not input_paths:
#         input_paths = [DEFAULT_INPUT_PATH]
#     if not output_paths:
#         output_paths = [DEFAULT_OUTPUT_PATH]
    
#     clear_output_folders(output_paths, project_root)
#     return activate_main(input_paths, output_paths, config_path, project_root)

def main():
    """Función main para compatibilidad con ejecución directa."""
    if len(sys.argv) == 1:
        input_paths = [DEFAULT_INPUT_PATH]
        output_paths = [DEFAULT_OUTPUT_PATH]
        config_path = DEFAULT_CONFIG_FILE
        project_root = PROJECT_ROOT
        
        clear_output_folders(output_paths, project_root)
        return activate_main(input_paths, output_paths, config_path, project_root)
    # app()

if __name__ == "__main__":
    main()