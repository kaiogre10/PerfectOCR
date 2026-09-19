# service/cache_manager.py
import shutil
import os
import logging
import platform
from typing import Set, Tuple, Optional, List, Dict, Any
#from psycopg2 import sql
from services.log_service import basic_exc_logger
from core.assets.patterns import extension_suffix
from core.assets.assets import VALID_IMG_EXT, INVALID_EXTENSIONS, TRASH_EXT, CACHE_DIRS, EXCLUDED_DIRS, NO_DEL, ALL_FILES_TYPES
from domain.class_models import OSModels

_extension_suffix = extension_suffix
PROJECT_ROOT: str = ""
output_paths: List[str] = []

valid_img_ext = VALID_IMG_EXT
invalid_extensions = INVALID_EXTENSIONS
trash_ext = TRASH_EXT
cache_dirs = CACHE_DIRS
excluded_dirs = EXCLUDED_DIRS
no_del = NO_DEL
all_files_types = ALL_FILES_TYPES

SO = {}

logger = logging.getLogger(__name__)

def init_system(project_root: str):
    global PROJECT_ROOT, SO
    PROJECT_ROOT = project_root  # type: ignore
    SO = get_so()

def set_system_config(config: Dict[str, List[str]]):
    global output_paths
    if config:
        output_paths = config["output_paths"]
        # output_paths = [os.path.join(PROJECT_ROOT, folder) for folder in output_path]

def _can_delete_entry(path: str) -> bool:
    """
    Verifica permisos mínimos de borrado:
    Para borrar un archivo/carpeta se requiere permiso de escritura + ejecución
    en su carpeta padre.
    """
    if path.endswith(no_del):
        return False

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

def clear_output_folders():
    """Vacia carpetas de salida de forma segura.
    - Si falta permiso en cualquier objetivo, no elimina nada.
    - Carpetas: se eliminan completas.
    - Archivos sueltos: solo extensiones objetivo.
    """
    if not output_paths:
        logger.info(f"NO HAY ARCHIVOS OUTPUT, NO SE LIMPIARÁ NADA")
        return
        
    deleted_files = 0
    deleted_folder = 0
    ok, blocked = _preflight_delete_plan(output_paths, all_files_types)
    if not ok:
        logger.warning("Limpieza abortada: hay rutas sin permisos. No se eliminó nada.")
        for p in blocked:
            logger.warning(f"Sin permisos: '{p}'")
            continue

    logger.debug("Limpieza Inicial: Vaciando carpetas de salida")
    for folder_path in (output_paths or cache_dirs):
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
                    logger.warning(f"Carpeta eliminada: {item_path}")
                else:
                    ext = os.path.splitext(item_name)[1].lower()
                    if ext in all_files_types:
                        os.remove(item_path)
                        deleted_files += 1
                        logger.info(f"Archivo eliminado: {item_path}")
                    else:
                        logger.debug(f"Saltado por extensión no permitida: {item_path}")

            except OSError as e:
                basic_exc_logger(f"Error al eliminar {item_path}: {e}", exc_info=True)
                
    logger.info(f"Archivos eliminados: {deleted_files}, Carpetas eliminadas: {deleted_folder}")

def cleanup_project_cache(specific_files: Optional[List[str]] = None, aditional_dirs: Optional[List[str]] = None):
    """Elimina la caché y residuos del proyecto """
    if aditional_dirs is not None and aditional_dirs:
        aditional_dirs = [path for path in aditional_dirs if os.path.isdir(path)]
        cache_dirs.extend(aditional_dirs)
    
    if specific_files is not None and specific_files:
        specific_set: Set[str] = set(path for path in specific_files if os.path.isfile(path))
        if specific_set:
            specific_files_set: Tuple[str, ...] = tuple(os.path.basename(path) for path in specific_set)
        else:
            specific_files_set = tuple()
    else:
        specific_files_set = tuple()
        specific_set = set()

    try:
        for dirpath, dirnames, filenames in os.walk(PROJECT_ROOT):
            for ed in excluded_dirs:
                if ed in dirnames:
                    #ex_cache_path = os.path.join(dirpath, ed)
                    #basic_exc_logger(f"DIRECTORIO OMITIDO: {ex_cache_path}")
                    dirnames.remove(ed)

            for d in dirnames:
                if d in cache_dirs:
                    try:
                        cache_path = os.path.join(dirpath, d)
                        shutil.rmtree(cache_path)
                        dirnames.remove(d)
                        # basic_exc_logger(f"DIRECTORIO ELIMINADO: {cache_path}")
                    except FileNotFoundError as e:
                        basic_exc_logger(f"Error al eliminar '{cache_path}': {e}", exc_info=True) # type: ignore
                        continue
            try:
                for filename in filenames:
                    if filename.endswith(trash_ext) or (False if not specific_files_set else (filename in specific_files_set or filename.endswith(SO))):
                        file_path: str = os.path.join(dirpath, filename)
                        os.remove(file_path)
                        if file_path in specific_set:
                            specific_set.remove(file_path)
                            basic_exc_logger(f"ARCHIVO TARGET ELIMINADO: '{file_path}'")
                            continue

                        basic_exc_logger(f"Eliminado archivo de caché: '{file_path}'")
                        continue

            except FileNotFoundError as e:
                basic_exc_logger(f"Error eliminando '{filenames}': {e}", exc_info=True)
                raise

    except FileNotFoundError as e:
        basic_exc_logger(f"Error al eliminar {e}", exc_info=True)
        return

def clean_db(get_local_connection: Any) -> bool:
    """Limpia toda la db en postgre de manera automatizada para facilitar el testeo"""
    try:
        with get_local_connection as conn:
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

                truncate_query = sql.SQL("TRUNCATE TABLE OR RESTART IDENTITY CASCADE").format(
                    sql.SQL(", ").join(sql.Identifier(t) for t in tables)
                )
                cur.execute(truncate_query)
            conn.commit()

        logger.info("DB vaciada correctamente. Tablas truncadas: %s", ", ".join(tables))
        return True

    except Exception as e:
        logger.error("Error limpiando la DB: %s", e, exc_info=True)
        return False

def count_and_plan(config: Dict[str, Any]) -> List[str]:
    """
    PLANIFICA el procesamiento: cuenta imágenes y decide estrategia según las reglas:
    1. Si se especifican `images_names`, se buscan prioritariamente.
    2. Si no, se procesan todas las imágenes en `input_dirs`.
    3. Si se encuentran todos los `images_names` y quedan directorios, se procesan completos.
    """
    input_paths = config['input_dirs']
    skip_names = config.get("skip_names", {})
    names_to_find: Set[str] = config['images_names']
    if not names_to_find:
        if not input_paths:
            logger.error("No se proporcionaron rutas de entrada (input_dirs)")
            return []
    
    if skip_names:
        names_to_find.difference_update(skip_names)
        
    image_info: List[str] = []
    total_paths = len(input_paths)
    
    for i, path in enumerate(input_paths):
        if names_to_find:
            files_in_dir = get_images_in_dir(path, list(names_to_find))
            
            if files_in_dir:
                files_to_remove: Set[str] = set(files_in_dir)
                if not names_to_find.isdisjoint(files_to_remove):
                    names_to_find.difference_update(files_to_remove)

                for file in files_in_dir:
                    full_path = os.path.join(PROJECT_ROOT, path, file)
                    image_info.append(full_path)
                    continue

            elif names_to_find:
                continue

        elif total_paths >= i:
            all_files_dir = get_images_in_dir(path, [])
            if not all_files_dir:
                continue
            
            if skip_names:
                all_files_dir = [file for file in all_files_dir if _extension_suffix.sub("", file) not in skip_names]
                if not all_files_dir:
                    continue
                
            for file in all_files_dir:
                full_path = os.path.join(PROJECT_ROOT, path, file)
                image_info.append(full_path)
                continue
        else:
            break

    if not image_info:
        raise FileNotFoundError(f"No se encontraron imágenes válidas en las rutas especificadas: '{input_paths}'")

    basic_exc_logger(f"'{len(image_info)}' IMAGENES PARA PROCESAR")
    return image_info

def get_images_in_dir(input_path: str, files_to_find: List[str]) -> List[str]:
    """Devuelve todos los archivos que haya en un directorio si son una extensión válida, si no se entrega lista de archivos devuelve todo los archivos del directorio"""
    files_name_dir = [file for _, _, files in os.walk(input_path) for file in files if os.path.splitext(file)[1] in valid_img_ext]
    if not files_name_dir:
        return []
    if not files_to_find:
        return files_name_dir

    split_names = [os.path.splitext(file) for file in files_name_dir]
    files_in_dir = ["".join(name) for name in split_names if name[0] in files_to_find]
    return files_name_dir if not files_in_dir else files_in_dir

def get_so() -> Tuple[str, ...]:
    """[0]Bibliotecas, [1]Librerias"""
    global SO
    SO = platform.system()
    logger.warning(f"'{OSModels.LINUX.name.capitalize()}' -> {SO}")
    if SO == OSModels.WINDOWS.name.capitalize:
        return OSModels.WINDOWS.value

    elif SO == OSModels.LINUX.name.capitalize():
        val = OSModels.LINUX.value
        logger.warning(f"' VAL: {val}'")
        return val
    else:
        return SO
    
def cleanup_project(specific_files: Optional[List[str]] = None, aditional_dirs: Optional[List[str]] = None):
    clear_output_folders()
    cleanup_project_cache(specific_files, aditional_dirs)