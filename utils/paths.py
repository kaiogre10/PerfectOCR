import os
from typing import List, Optional, Tuple
import logging
from core.assets.patterns import extension_suffix
from collections.abc import Iterable
from core.assets.assets import KIND_SEARCH_LIST, KindofSearch

filename_search = KIND_SEARCH_LIST[0]
no_extensions_search  = KIND_SEARCH_LIST[1]
extensions_search = KIND_SEARCH_LIST[2]
prefix_search = KIND_SEARCH_LIST[3]
subfix_search = KIND_SEARCH_LIST[4]

_extension_suffix = extension_suffix
project_root: Optional[str] = None

logger = logging.getLogger(__name__)

def set_project_root(PROJECT_ROOT: str):
    global project_root
    project_root = PROJECT_ROOT

def get_all(search_root: str, skip_names: Optional[Iterable[str]] = None) -> Tuple[List[str], List[str]]:
    """"Devuele una Tupla de listas (carpetas[0], archivos[1]) con las rutas relativas a search_root de todos los archivos contenidos, sirve solo como exploración local, no entrega rutas absolutas, temporalmente solo omite directorios"""
    if not os.path.isdir(search_root):
        raise NotADirectoryError(f"SEARCH ROOT TIENE QUE SER UN CARPETA O DIRECTORIO, NO UN ARCHIVO: '{search_root}'")
    
    all_dirs: List[str] = []
    all_files: List[str] = []
    for dirpath, dirnames, filenames in os.walk(search_root):
        if skip_names is not None and skip_names:
            dirnames[:] = [d for d in dirnames if d not in skip_names]

        for d in dirnames:
            folders = os.path.join(dirpath, d)
            all_dirs.append(folders)

        for filename in filenames:
            
            files = os.path.join(dirpath, filename)
            all_files.append(files)

    return all_dirs, all_files

def build_from_dir(objetives: Iterable[str], search_root: Optional[str] = None, parent_include: Optional[bool] = None, parent: Optional[List[str]] | None = None, kind_search: KindofSearch | None = None, skip_names: Optional[Iterable[str]] = None) -> List[str]:
    """
    Busca objetivos especificos en el sistema de carpetas del proyecto, dentro de un marco delimitado y arbitrario.
    ``NONE`` ES UN RESULTADO VALIDO Y REPRESENTA UN ESTADO SEPARADO DE FALSE Y TRUE.
    Distringue entre busqueda de directorios, archivos por nombre con/sin extension, nombres prefijos, *subfijos (nombre sin extesion), referencias (nombres incompletos, *fuzzy) y formato.
    *están en fase Beta.

    Args:
        ``objetives`` (Iterable[str]): Lista o Tupla de nombres solicitados para buscar, pueden ser archivos, carpetas, extensiones o referencias.
        ``search_root`` (str | None, optional): Path completo de la carpeta padre que sirve como delimitador de búsqueda y simula ser el project_root, ambos tienen exactamente el mismo comportamiento, tiene priotidad sobre la raiz no pueden usarase simultaneamente.
        ``parent_include`` (bool | None, optional): Indica si devolver la carpeta con la jerarquía más alta en la cual existe una ruta valida para cada objetive.
        ``parent`` (list[str] | None, optional): Representación de un path completo que parte desde dir_scoope en forma de una lista sepadarada en jerarquia.
        ``kind_search`` (KindSearch | None, optional): Indicador especial que define el comportamiento y tipo de busqueda para los objetives:.
            ``None`` (None): Tipo de busqueda valida, y general que el motor infiere según el resto de los parametro solicitados.
            ``filename`` (str, filename): aplica fullmatch de nombres.
            ``no_extensions``:
            ``extensions``: busqueda por extensiones de archivos, pueden ser puras extesiones o archivos con extensiones.
            ``prefix``: Busqieda por referencia del inicio del nombre.
            ``subfix``: Busqieda por referencia del finel del nombre SIN FORMATO/EXTENSION.

        ``skip_names``(Iterable[str] | None, optional): Lista o Tupla de excepciones en la búsqueda, su significado semantico depende completamente de kind_search, es disjunto de objetives.
        
        Los resultados se devuelven en orden inverso al recorrido de os.walk.
        Si parent_include es verdadero, la ruta del directorio de búsqueda se agrega al final.
        Por ello, las llamadas que desempaquetan posiciones concretas deben conservar el orden de objetives y la estructura esperada del árbol.

        Si la ruta inicial no existe, los objetivos están vacíos o la combinación de filtros es inválida, registra una advertencia y devuelve una lista vacia.

    Returns:
        list: Lista de direcciones absoluta de cada obejtive encontrado
    """
    if not objetives:
        logger.warning("SIN OBJETIVOS")
        return []
    
    elif skip_names is not None and skip_names:
        skip_names: Tuple[str, ...] = tuple(skip_names)

        if not all(isinstance(x, (list, tuple)) for x in (objetives, skip_names)):
            raise TypeError("Todos los argumentos deben ser list o tuple")

        elif not all(names not in objetives for names in skip_names):
            raise ValueError(f"REPETICIONES ENTRE OMISIONES Y BUSQUEDAS")

        elif not isinstance(objetives, tuple):
            objetives: Tuple[str, ...] = tuple(objetives)
            skip_names: Tuple[str, ...] = (skip_names)

    elif not isinstance(objetives, list | tuple):
        raise ValueError("ERROR EN OBJETIVES")

    else:
        objetives: Tuple[str, ...] = tuple(objetives)
        skip_names: Tuple[str, ...] = tuple("")

    if project_root is None and search_root is None:
        raise TypeError("AMBOS ROOTS SON NONE")

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
        raise ValueError(f"RUTA INVALIDA: {search_dir}")
        
    search_type = _call_kind_and_data_search(kind_search)
    
    results = _locate_and_index_targets(search_dir, search_type, objetives, skip_names)
    
    if parent_include is not None and parent_include:
        results.append(search_dir)
    return results

def _locate_and_index_targets(search_dir: str, search_type: Tuple[Optional[bool], Optional[str]], objetives: Tuple[str, ...], skip_names: Tuple[str, ...]) -> List[str]:
    skip_paf, skip_ext, extensions = _get_exceptions_state(objetives, skip_names, search_type[0])
    if skip_paf and skip_ext:
        logger.warning("MISMO ESTAOD")
        return []

    valid_dirs: List[str] = []
    referencial_search = search_type[1]
    
    for dirpath, dirnames, filenames in os.walk(search_dir):
        if not extensions:
            # if skip_paf:
            #     dirnames[:] = [d for d in dirnames if d not in skip_names]
            for dir in skip_names:
                if skip_paf and dir in dirnames:
                    dirnames.remove(dir)
                    continue

            for d in dirnames:
                if d in objetives:
                    valid_dirs.append(os.path.join(dirpath, d))

        for filename in filenames:
            if referencial_search:
                if referencial_search == prefix_search and filename.startswith(objetives):
                    valid_dirs.append(os.path.join(dirpath, filename))
                    continue
                    
                # elif referencial_search == subfix_search and filename.endswith(_extension_suffix.sub(objetives, "")):
                #     valid_dirs.append(os.path.join(dirpath, filename))
                #     continue

            if skip_paf and filename in skip_names:
                filenames.remove(filename)
                continue

            if skip_ext and filename.endswith(skip_names):  # extensiones, diferentes de los subfijos
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
    return valid_dirs

def _get_exceptions_state(objetive: Tuple[str, ...], skip_name: Tuple[str, ...], with_extensions: Optional[bool] = None) -> Tuple[bool, bool, Optional[bool]]:
    """
    Decide si hay que filtrar archivos si hay que hacerlo cuales filtrar. Returns: (skip_paf, skip_ext, extensions)
    extensions: 
    """
    extensions: Optional[bool] = (with_extensions is not None and with_extensions) and all(_extension_suffix.fullmatch(ext) for ext in objetive)  # Asegurar que las extensiones sean validas, objetives pasa a ser formato de documento
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

def _call_kind_and_data_search(kind_search: Optional[str] = None) -> Tuple[Optional[bool], Optional[str]]:
    with_extensions: Optional[bool] = None
    referencial_search: Optional[str] = ""

    if kind_search is None:         # La orden para extensiones no existe, por lo que hay que inferir, ya que: (None == False) es incorrecto, hay que inferir el tipo de busqueda
        pass
    
    elif kind_search == extensions_search:  # Aquí se solicitaron extensiones explicitamente, entonces el tipo de busqueda es por extension de archivo
        with_extensions = True

    elif kind_search == no_extensions_search:  # Aquí las las no extensions implican la busqueda por nombre completo de carpeta o archivo
        with_extensions = False

    elif kind_search == filename_search:       # Explicitamente se solicito busqueda de archivos (todavia se analiza la posibilidad con/sin extensiones)
        with_extensions = False
    
    elif kind_search == prefix_search:  # Se solicitó un tipo de busqueda especial por referencias de nombres incompletas
        with_extensions = False
        referencial_search = prefix_search
    
    elif kind_search == subfix_search:
        referencial_search = subfix_search      # Extensions tiene que ser none para no confundir el nombre

    else:
        raise ValueError(f"ERROR EN EL TIPO DE BUSQUEDA: '{kind_search}', RETORONANDO")  # Tipo de busqueda inválida
    
    return with_extensions, referencial_search