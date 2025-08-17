# perfectocr/management/cache_manager.py
import shutil
import time
import os
import logging
from typing import List

logger = logging.getLogger(__name__)

def clear_output_folders(output_paths: List[str], project_root: str) -> None:
    """Vacia las carpetas de salida definidas en la config y cuenta los eliminados."""
    archivos_eliminados = 0
    carpetas_eliminadas = 0

    t0 = time.perf_counter()
    logger.debug("Limpieza Inicial: Vaciando carpetas de salida")
    for folder_path in output_paths:
        if not os.path.isdir(folder_path):
            continue
        for item_name in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item_name)
            try:
                if os.path.isdir(item_path):
                    # Contar archivos y carpetas dentro antes de eliminar
                    for root, dirs, files in os.walk(item_path):
                        carpetas_eliminadas += len(dirs)
                        archivos_eliminados += len(files)
                    shutil.rmtree(item_path)
                    carpetas_eliminadas += 1  # la carpeta principal
                else:
                    os.remove(item_path)
                    archivos_eliminados += 1
                logger.debug(f"Eliminado: {item_path}")
            except Exception as e:
                logger.error(f"Error al eliminar {item_path}: {e}")
    tempo = time.perf_counter() - t0
    total_eliminados = archivos_eliminados + carpetas_eliminadas
    avg_time_file = tempo / archivos_eliminados
    logging.info(
        f"Limpieza inicial completada en {tempo:.6f}s. "
        f"Archivos eliminados: {archivos_eliminados}, Carpetas eliminadas: {carpetas_eliminadas}, "
        f"Total: {total_eliminados}, promedio por archivo {avg_time_file:.6f} archivos/s"
    )

def cleanup_project_cache(project_root: str) -> None:
    """Elimina la caché del proyecto (__pycache__ y .pyc)."""
    project_root = project_root
    t0 = time.perf_counter()
    logger.debug(" Limpieza Final: Eliminando caché del proyecto")
    cache_path: str
    for dirpath, dirnames, filenames in os.walk(project_root):
        for d in list(dirnames):
            if d == "__pycache__":
                try:
                    cache_path = os.path.join(dirpath, d)
                    shutil.rmtree(cache_path)
                    logger.debug(f"Eliminada carpeta de caché: {cache_path}")
                    dirnames.remove(d)
                    
                except Exception as e:
                    logger.error(f"Error al eliminar {cache_path} {e}")
                    return
        
        # Eliminar archivos .pyc y .pyo
        filename: str
        file_path: str
        for filename in filenames:
            if filename.endswith(('.pyc', '.pyo')):
                try:
                    file_path = os.path.join(dirpath, filename)
                    os.remove(file_path)
                    logger.debug(f"Eliminado archivo de caché: {file_path}")
                except Exception as e:
                    logger.error(f"Error al eliminar {file_path}: {e}")
                    return
    tempo = time.perf_counter() - t0
    logger.info(f"Limpieza de cache completada en {tempo}")    
