import os
from typing import List, Optional, Tuple
import logging
from core.assets.patterns import extension_suffix
from collections.abc import Iterable

kind_search_list = ["filename", "no_extensions", "extensions", "reference"]

filename_search: str = kind_search_list[0]
no_extensions_search: str = kind_search_list[1]
extensions_search: str = kind_search_list[2]
reference_search: str = kind_search_list[3]

_extension_suffix = extension_suffix
project_root: Optional[str] = None

logger = logging.getLogger(__name__)

def set_project_root(PROJECT_ROOT: str):
    global project_root
    project_root = PROJECT_ROOT

def build_from_dir(objetives: Iterable[str], search_root: Optional[str] = None, parent_include: Optional[bool] = None, parent: Optional[List[str]] = None, kind_search: Optional[str] = None, skip_names: Optional[Iterable[str]] = None) -> List[str]:
    """
    Busca recursivamente archivos o carpetas y devuelve sus rutas completas:
    Objetives: Lista de nombres solicitados para buscar, pueden ser archivos, carpetas, extensiones o referencias
    Search_root: Path completo de la carpeta padre que sirve como delimitador de búsqueda y simula ser el project_root, ambos tienen exactamente el mismo comportamiento, tiene priotidad sobre project_root y no puede usarse al mismo tiempo que project root
    Parent_include: Indica si devolver la carpeta con la jerarquía más alta en la cual existe una ruta valida para cada Objetive
    Parent: Representación de un path completo que parte desde dir_scoope en forma de una lista sepadarada en jerarquia
    Kind_search: Indicador especial que define el comportamiento y tipo de busqueda para los Objetives:
        None o 'filename': aplica fullmatch de nombres
        'extensions': busqueda por extensiones de archivos, pueden ser puras extesiones o archivos con extensiones
        'reference': Busqueda por referencia de nombre, tipo fuzzy/regex o matcheos internos
        'prefix': Por inicio del nombre

    Skip_names: Cojunto de excepciones en la búsqueda, su significado semantico depende completamente de Kind_search, es disjunto de Objetives

    La búsqueda parte de un ``dir_scoope`` definido cuando search_root es una dirección valida y ``parent`` es ``None``;
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
        logger.warning("SIN OBJETIVOS")
        return []
    
    elif skip_names is not None and skip_names:
        skip_names: Tuple[str, ...] = tuple(skip_names)

        if not all(isinstance(x, (list, tuple)) for x in (objetives, skip_names)):
            raise TypeError("Todos los argumentos deben ser list o tuple")

        elif not set(skip_names).isdisjoint(objetives):
            raise ValueError(f"REPETICIONES ENTRE OMISIONES Y BUSQUEDAS: {set(skip_names).intersection(set(objetives))}")

        elif not isinstance(objetives, tuple):
            objetives: Tuple[str, ...] = tuple(objetives)
            skip_names: Tuple[str, ...] = (skip_names)

    elif not isinstance(objetives, list | tuple):
        raise ValueError("ERROR EN OBJETIVES")

    else:
        objetives: Tuple[str, ...] = tuple(objetives)
        skip_names: Tuple[str, ...] = tuple("")

    if project_root is None and search_root is None:
        raise TypeError(f"AMBOS ROOTS SON NONE")

    elif search_root is None:
        dir_scoope = project_root
    else:
        dir_scoope = search_root

    if not dir_scoope:
        raise TypeError("ERROR CON LOS ROOTS")

    if parent is None:
        search_dir = dir_scoope
    else:
        search_dir = os.path.join(dir_scoope, *parent)

    if not os.path.isdir(search_dir):
        logger.warning(f"RUTA INVALIDA: {search_dir}")
        return []
    
    logger.warning(f"SEARCH DIR: {search_dir}")
    referencial_search = (kind_search == reference_search)

    if kind_search is None:
        with_extensions = None     # Mantenemos la misma logica de la implementación anterior, donde None significaba sin extensiones==archivos y directorios
    
    elif kind_search == extensions_search:
        with_extensions = True

    elif kind_search == no_extensions_search:
        with_extensions = False

    elif kind_search == filename_search:
        with_extensions = False

    elif referencial_search:
        with_extensions = False

    else:
        raise ValueError(f"ERROR EN EL TIPO DE BUSQUEDA, RETORONANDO")

    skip_paf, skip_ext, extensions = _get_exceptions_state(objetives, skip_names, with_extensions)
    if skip_paf and skip_ext:
        logger.warning("MISMO ESTAOD")
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
            if referencial_search:
                if filename.startswith(objetives):
                    valid_dirs.append(os.path.join(dirpath, filename))
                continue

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

def _get_exceptions_state(objetive: Tuple[str, ...], skip_name: Tuple[str, ...], with_extensions: Optional[bool] = None):
    """Decide si hay que filtrar archivos si hay que hacerlo cuales filtrar. Returns: (skip_paf, skip_ext, extensions)"""
    extensions = (with_extensions is not None and with_extensions) and all(_extension_suffix.fullmatch(ext) for ext in objetive)  # Asegurar que las extensiones sean validas, objetives pasa a ser formato de documento
    if skip_name:       # Si hay que hacer excepciones paths and files 'paf'
        skip = all(_extension_suffix.fullmatch(ext) for ext in skip_name)  # Si las excepciones de extensiones son validas, pasa a ser formato de documento.
        if extensions:               # Hay que trabajar con extensiones, esta validación esconde la verdadera validacion 'if extensions is not None'
            if not extensions:       # Aqui ya se valido que extensions existe
                if not skip:
                    skip_paf = True
                    skip_ext = True
                elif not all(ext.isalpha() for ext in skip_name):
                    skip_paf = True
                    skip_ext = True
                else:
                    logger.warning("TODAS ALFABETICAS")
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