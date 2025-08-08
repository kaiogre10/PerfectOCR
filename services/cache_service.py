# perfectocr/management/cache_manager.py
import shutil
import os
import logging
from typing import List

logger = logging.getLogger(__name__)

def clear_output_folders(output_paths: List[str], project_root: str) -> None:
    """Vacia las carpetas de salida definidas en la config."""
    project_root = project_root
    folders_to_empty = output_paths
    
    logger.info("--- Limpieza Inicial: Vaciando carpetas de salida ---")
    for folder_path in folders_to_empty:
        if not os.path.isdir(folder_path):
            continue
        for item_name in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item_name)
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
                logger.debug(f"Eliminado: {item_path}")
            except Exception as e:
                logger.error(f"Error al eliminar {item_path}: {e}")

def cleanup_project_cache(project_root: str) -> None:
    """Elimina la caché del proyecto (__pycache__ y .pyc)."""
    project_root = project_root
    logger.info("--- Limpieza Final: Eliminando caché del proyecto ---")
    
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
        for filename in filenames:
            if filename.endswith(('.pyc', '.pyo')):
                try:
                    file_path = os.path.join(dirpath, filename)
                    os.remove(file_path)
                    logger.debug(f"Eliminado archivo de caché: {file_path}")
                except Exception as e:
                    logger.error(f"Error al eliminar {file_path}: {e}")
                    return
                    