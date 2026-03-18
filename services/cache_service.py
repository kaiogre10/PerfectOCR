# service/cache_manager.py
import shutil
import os
import logging
from typing import List, Set, Tuple

logger = logging.getLogger(__name__)

DEFAULT_ALLOWED_EXTENSIONS: Set[str] = {
    ".json", ".txt",
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"
}

def _can_delete_entry(path: str) -> bool:
    """
    Verifica permisos mínimos de borrado:
    - Para borrar un archivo/carpeta se requiere permiso de escritura + ejecución
      en su carpeta padre.
    """
    parent = os.path.dirname(path) or "."
    return os.access(parent, os.W_OK | os.X_OK)

def _preflight_delete_plan(output_paths: List[str], exts: Set[str]) -> Tuple[bool, List[str]]:
    """
    Fase 1 (sin borrar): arma plan y valida permisos.
    Si falta permiso en cualquier entrada, devuelve False y lista de bloqueos.
    """
    blocked: List[str] = []

    for folder_path in output_paths:
        if not os.path.isdir(folder_path):
            continue

        # Necesario para listar contenido
        if not os.access(folder_path, os.R_OK | os.X_OK):
            blocked.append(folder_path)
            continue

        for item_name in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item_name)

            if os.path.isdir(item_path):
                # Se borra carpeta completa -> validar borrado de esa entrada
                if not _can_delete_entry(item_path):
                    blocked.append(item_path)
                    continue

                # Validación extra para recorrer sin errores de permisos
                for root, _, _ in os.walk(item_path):
                    if not os.access(root, os.R_OK | os.X_OK):
                        blocked.append(root)
                        break
            else:
                ext = os.path.splitext(item_name)[1].lower()
                if ext in exts and not _can_delete_entry(item_path):
                    blocked.append(item_path)

    return (len(blocked) == 0, blocked)

def clear_output_folders(output_paths: List[str]) -> None:
    """Vacia carpetas de salida de forma segura.
    - Si falta permiso en cualquier objetivo, no elimina nada.
    - Carpetas: se eliminan completas.
    - Archivos sueltos: solo extensiones objetivo.
    """
    deleted_files = 0
    deleted_folder = 0
    exts = {e.lower() for e in DEFAULT_ALLOWED_EXTENSIONS}

    # Fase 1: preflight (fail-closed)
    ok, blocked = _preflight_delete_plan(output_paths, exts)
    if not ok:
        logger.error("Limpieza abortada: hay rutas sin permisos. No se eliminó nada.")
        for p in blocked:
            logger.error(f"Sin permisos: {p}")
        return

    # Fase 2: ejecución real (solo si todo pasó preflight)
    logger.debug("Limpieza Inicial: Vaciando carpetas de salida")
    for folder_path in output_paths:
        if not os.path.isdir(folder_path):
            continue

        for item_name in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item_name)
            try:
                if os.path.isdir(item_path):
                    for _, dirs, files in os.walk(item_path):
                        deleted_folder += len(dirs)
                        deleted_files += len(files)

                    shutil.rmtree(item_path)
                    deleted_folder += 1
                    logger.debug(f"Carpeta eliminada: {item_path}")
                else:
                    ext = os.path.splitext(item_name)[1].lower()
                    if ext in exts:
                        os.remove(item_path)
                        deleted_files += 1
                        logger.debug(f"Archivo eliminado: {item_path}")
                    else:
                        logger.debug(f"Saltado por extensión no permitida: {item_path}")

            except Exception as e:
                # Defensa adicional
                logger.error(f"Error al eliminar {item_path}: {e}", exc_info=True)
                logger.error("Se detiene la limpieza por seguridad.")
                return

    logger.debug(f"Archivos eliminados: {deleted_files}, Carpetas eliminadas: {deleted_folder}")

def cleanup_project_cache(project_root: str):
    """Elimina la caché del proyecto (__pycache__ y .pyc)."""
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
