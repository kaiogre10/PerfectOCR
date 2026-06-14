# service/cache_manager.py
import shutil
import os
import logging
import platform
from typing import Set, Tuple, Optional
from services.db_service import DataBaseService
from psycopg2 import sql
from typing import List, Dict, Any

PROJECT_ROOT: str = ""

def set_project_root(project_root: str):
    global PROJECT_ROOT
    PROJECT_ROOT = project_root # type: ignore

logger = logging.getLogger(__name__)

DEFAULT_ALLOWED_EXTENSIONS: Set[str] = {
    ".json", ".txt",
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"
}

trash_ext: Tuple[str, ...] = ('.pyc', '.pyo', ".c")

valid_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.pbm', '.pgm', '.ppm', '.jp2')

def _can_delete_entry(path: str) -> bool:
    """
    Verifica permisos mínimos de borrado:
    Para borrar un archivo/carpeta se requiere permiso de escritura + ejecución
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
                return

    logger.debug(f"Archivos eliminados: {deleted_files}, Carpetas eliminadas: {deleted_folder}")

def cleanup_project_cache(aditional_files: Optional[str] = None):
    """Elimina la caché y residuos del proyecto """
    cache_path: str
    try:
        for dirpath, dirnames, filenames in os.walk(PROJECT_ROOT):
            for d in list(dirnames):
                if d in ("__pycache__", "build", "dist"):
                    try:
                        cache_path = os.path.join(dirpath, d)
                        shutil.rmtree(cache_path)
                        dirnames.remove(d)
                        
                    except Exception as e:
                        logger.error(f"Error al eliminar {cache_path}: {e}") # type: ignore
                        return
            
            filename: str
            file_path: str
            if aditional_files is not None:
                trash_extensions: Tuple[str, ...] = trash_ext + tuple(aditional_files.split(','))
            else:
                trash_extensions = trash_ext
                
            for filename in filenames:
                if filename.endswith(trash_extensions):
                    file_path = os.path.join(dirpath, filename)
                    os.remove(file_path)
                    logger.debug(f"Eliminado archivo de caché: {file_path}")
                        
    except Exception as e:
        logger.error(f"Error al eliminar {file_path}: {e}") # type: ignore
        return

def clean_db(db_service: DataBaseService) -> bool:
    try:
        with db_service.get_connection() as conn:
            with conn.cursor() as cur:
                # Trae todas las tablas reales del esquema public
                cur.execute("""
                    SELECT tablename
                    FROM pg_tables
                    WHERE schemaname = 'public'
                    ORDER BY tablename;
                """)
                tables = [row[0] for row in cur.fetchall()]

                if not tables:
                    logger.warning("No hay tablas para truncar en schema public")
                    conn.commit()
                    return True

                # TRUNCATE TABLE t1, t2, ... RESTART IDENTITY CASCADE
                truncate_query = sql.SQL("TRUNCATE TABLE {} RESTART IDENTITY CASCADE").format(
                    sql.SQL(", ").join(sql.Identifier(t) for t in tables)
                )
                cur.execute(truncate_query)
            conn.commit()

        logger.info("DB vaciada correctamente. Tablas truncadas: %s", ", ".join(tables))
        return True

    except Exception as e:
        logger.error("Error limpiando la DB: %s", e, exc_info=True)
        return False

def count_and_plan(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    PLANIFICA el procesamiento: cuenta imágenes y decide estrategia según las reglas:
    1. Si se especifican `images_names`, se buscan prioritariamente.
    2. Si no, se procesan todas las imágenes en `input_dirs`.
    3. Si se encuentran todos los `images_names` y quedan directorios, se procesan completos.
    """
    input_paths: List[str] = config['input_dirs']
    images_names = config['images_names']
    if not input_paths:
        logger.warning("No se proporcionaron rutas de entrada (input_dirs).")
        return {}

    image_info: List[Dict[str, Any]] = []
    names_to_find = set(images_names)
    total_paths = len(input_paths)
    
    for i, path in enumerate(input_paths):
        if names_to_find:
            files_in_dir = get_images_in_dir(path, list(names_to_find))
            if files_in_dir:
                files_to_remove = set(files_in_dir)
                if not names_to_find.isdisjoint(files_to_remove):
                    names_to_find.discard(files_to_remove)
                
                for file in files_in_dir:
                    full_path = os.path.join(PROJECT_ROOT, path, file)
                    image_info.append({"full_path": full_path, "name": file})
                    continue
                
            elif names_to_find:
                continue
            
        elif total_paths >= i:
            all_files_dir = get_images_in_dir(path, [])
            if not all_files_dir:
                continue

            for file in all_files_dir:
                full_path = os.path.join(PROJECT_ROOT, path, file)
                image_info.append({"full_path": full_path, "name": file})
                continue
        else:
            break

    if not image_info:
        logger.error("No se encontraron imágenes válidas en las rutas especificadas.")
        return {}
        
    return {"image_info": image_info}
    
def get_images_in_dir(input_path: str, files_list: List[str]) -> List[str]:
    files_name_dir = [file for _, _, files in os.walk(input_path) for file in files if file.endswith(valid_extensions)]
    if not files_name_dir:
        return []
    if not files_list:
        return files_name_dir

    split_names = [os.path.splitext(file) for file in files_name_dir]        
    files_in_dir = ["".join(name) for name in split_names if name[0] in files_list]
    # logger.info(f"INTER IDX: {files_in_dir}")
    return files_name_dir if not files_in_dir else files_in_dir

def get_so() -> str:
    if platform.system() == "Windows":
        return ".dll"
    elif platform.system() == "Linux":
        return ".so"
    else:
        # MacOS
        return ".dylib"