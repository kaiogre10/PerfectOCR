# service/cache_manager.py
import shutil
import os
import logging
from typing import List

logger = logging.getLogger(__name__)

def clear_output_folders(output_paths: List[str]) -> None:
    """Vacia las carpetas de salida definidas en la config y cuenta los eliminados."""
    deleted_files = 0
    deleted_folder = 0

    logger.debug("Limpieza Inicial: Vaciando carpetas de salida")
    for folder_path in output_paths:
        if not os.path.isdir(folder_path):
            continue
        
        for item_name in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item_name)
            try:
                if os.path.isdir(item_path):
                    # Contar archivos y carpetas dentro antes de eliminar
                    for root, dirs, files in os.walk(item_path): # type: ignore
                        deleted_folder += len(dirs)
                        deleted_files += len(files)
                        
                    shutil.rmtree(item_path)
                    deleted_folder += 1  # la carpeta principal
                    
                else:
                    os.remove(item_path)
                    deleted_files += 1
                    
                logger.debug(f"Eliminado: {item_path}")
                
            except Exception as e:
                logger.error(f"Error al eliminar {item_path}: {e}", exc_info=True)
                
    total_eliminated = deleted_files + deleted_folder
    if deleted_files < 0:
        logger.debug(f"Total: {total_eliminated} archivos/s")
    else:
        pass        
        
    logger.debug(f"Archivos eliminados: {deleted_files}, Carpetas eliminadas: {deleted_folder}")

def cleanup_project_cache(project_root: str):
    """Elimina la caché del proyecto (__pycache__ y .pyc)."""
    project_root = project_root
    logger.debug(" Limpieza Final: Eliminando caché del proyecto")
    cache_path: str
    
    try:
        for dirpath, dirnames, filenames in os.walk(project_root):
            for d in list(dirnames):
                if d == "__pycache__":
                    
                    try:
                        cache_path = os.path.join(dirpath, d)
                        shutil.rmtree(cache_path)
                        dirnames.remove(d)
                        
                    except Exception as e:
                        logger.error(f"Error al eliminar {cache_path}: {e}") # type: ignore
                        return
            
            # Eliminar archivos .pyc y .pyo
            filename: str
            file_path: str
            for filename in filenames:
                if filename.endswith(('.pyc', '.pyo')):
                    file_path = os.path.join(dirpath, filename)
                    os.remove(file_path)
                    logger.debug(f"Eliminado archivo de caché: {file_path}")
                        
    except Exception as e:
        logger.error(f"Error al eliminar {file_path}: {e}") # type: ignore
        return
