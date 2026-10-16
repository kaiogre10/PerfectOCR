import os
from typing import List, Optional, Tuple
import logging
from core.assets.patterns import extension_suffix

_extension_suffix = extension_suffix
project_root: str = ""

logger = logging.getLogger(__name__)

def set_project_root(PROJECT_ROOT: str):
    global project_root
    project_root = PROJECT_ROOT

def build_from_dir(objetives: List[str], parent_include: Optional[bool] = None, parent: Optional[List[str]] = None, with_extensions: Optional[bool] = None, skip_names: Optional[List[str]] = None) -> List[str]:
    """
    Busca recursivamente archivos o directorios y devuelve sus rutas.

    La búsqueda parte de ``project_root`` cuando ``parent`` es ``None``;
    de lo contrario, ``parent`` se interpreta como una secuencia de
    componentes relativos a ``project_root``. Cada elemento de ``objetives``
    se compara por nombre exacto, excepto cuando ``with_extensions`` es
    verdadero y todos los objetivos tienen formato de extensión; en ese caso
    se comparan como sufijos de archivo, por ejemplo ``[".py"]``.

    Los resultados se devuelven en orden inverso al recorrido de
    ``os.walk``. Si ``parent_include`` es verdadero, la ruta del directorio
    de búsqueda se agrega al final. Por ello, las llamadas que desempaquetan
    posiciones concretas deben conservar el orden de ``objetives`` y la
    estructura esperada del árbol

    ``skip_names`` cambia el modo de exclusión según su contenido:

    * nombres de archivos o directorios: se excluyen durante el recorrido;
    * extensiones: se excluyen los archivos cuyo nombre termina con alguna
      de esas extensiones.

    Si la ruta inicial no existe, los objetivos están vacíos o la combinación
    de filtros es inválida, registra una advertencia y devuelve ``[]``.
    """
    if not objetives:
        logger.warning(f"SIN OBJETIVOS")
        return []

    if isinstance(objetives, list):
        objetives: Tuple[str, ...] = tuple(objetives)

    if parent is None:
        search_dir: str = project_root

    else:
        search_dir: str = os.path.join(project_root, *parent)

    if not os.path.isdir(search_dir):
        logger.warning(f"RUTA INVALIDA: {search_dir}")
        return []

    if skip_names is not None and skip_names:
        if isinstance(skip_names, list):
            skip_names: Tuple[str, ...] = tuple(skip_names)
    else:
        skip_names: Tuple[str, ...] = tuple()

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
    if parent_include is not None and parent_include:
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
                    logger.warning(f"TODAS ALFABETICAS")
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

def validate_files(paths: List[str]):
    return all(os.path.exists(dir) for dir in paths)
