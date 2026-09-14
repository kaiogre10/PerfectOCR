import os
from typing import List, Optional, Tuple
import logging
from core.assets.patterns import extension_suffix

logger = logging.getLogger(__name__)

_extension_suffix = extension_suffix
project_root: str = ""

def set_projet_root(PROJECT_ROOT: str):
    global project_root
    project_root = PROJECT_ROOT

def build_from_dir(objetives: List[str], parent_include: Optional[bool] = None, parent: Optional[List[str]] = None, with_extensions: Optional[bool] = None, skip_names: Optional[List[str]] = None) -> List[str]:
    """
    Busca recursivamente los objetivos dentro de parent y devuelve sus rutas. Asume que paren esta en project_root
    Si extensions es True, objectives se interpreta como extensiones de archivo.
    """
    if not objetives:
        logger.warning(f"SIN OBJETIVOS")
        return []

    if isinstance(objetives, list):
        objetives = tuple(objetives)

    if parent is None:
        search_dir: str = project_root
    else:
        search_dir: str = os.path.join(project_root, *parent)

    if not os.path.isdir(search_dir):
        logger.warning(f"RUTA INVALIDA: {search_dir}")
        return []

    if skip_names is not None and skip_names:
        if isinstance(skip_names, list):
            skip_names = tuple(skip_names)
    else:
        skip_names = tuple()

    skip_paf, skip_ext, extensions = _get_exceptions_state(objetives, skip_names, with_extensions)
    if skip_paf and skip_ext:
        logger.warning(f"MISMO ESTAOD")
        return []

    valid_dirs: List[str] = []
    for dirpath, dirnames, filenames in os.walk(search_dir):
        if not extensions:
            for dir in skip_names:
                if skip_paf and dir in dirnames:
                    dirnames.remove(dir)
                    continue

            for d in dirnames:
                if d in objetives:
                    valid_dirs.append(os.path.join(dirpath, d))

        for filename in filenames:
            if skip_paf and filename in skip_names:
                filenames.remove(filename)
                continue

            if skip_ext and filename.endswith(skip_names):
                filenames.remove(filename)
                continue

            if extensions:
                if filename.endswith(objetives):
                    valid_dirs.append(os.path.join(dirpath, filename))

            elif filename in objetives:
                valid_dirs.append(os.path.join(dirpath, filename))

            else:
                continue

    valid_dirs.reverse()
    if parent_include is not None and parent_include != project_root:
        valid_dirs.append(search_dir)
    return valid_dirs

def _get_exceptions_state(objetives: Tuple[str, ...], skip_names: Tuple[str, ...], with_extensions: Optional[bool] = None):
    """Decide si hay que filtrar archivos si hay que hacerlo cuales filtrar. Returns: (skip_paf, skip_ext, extensions)"""
    extensions = (with_extensions is not None and with_extensions) and all(_extension_suffix.fullmatch(ext) for ext in objetives)  # Asegurar que las extensiones sean validas, objetives pasa a ser formato de documento
    if skip_names:       # Si hay que hacer excepciones paths and files
        skip = all(_extension_suffix.fullmatch(ext) for ext in skip_names)  # Si las excepciones de extensiones son validas, pasa a ser formato de documento.
        if extensions:           # Hay que trabajar con extensiones
            if not extensions:
                if not skip:
                    skip_paf = True
                    skip_ext = True
                elif not all(ext.isalpha() for ext in skip_names):
                    skip_paf = True
                    skip_ext = True
                else:
                    logger.warning(f"TODAS ALFANETICAS")
                    skip_paf = False
                    skip_ext = True
                    extensions = True

            elif not skip:          # Aqui extensions es True
                skip_paf = False
                skip_ext = True
            else:
                skip_paf = False      # Aquí ambos valores pasan a ser False porque no se pueden filtrar extenciones ni archivos al mismo tiempo
                skip_ext = False
                extensions = True
        else:
            skip_paf = True
            skip_ext = False

    else:
        skip_paf = False
        skip_ext = False

    return skip_paf, skip_ext, extensions