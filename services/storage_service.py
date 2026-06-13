from typing import List, Any, Dict
import logging
import os
import ctypes

PROJECT_ROOT: str = ""
OUTPUT_PATHS: List[str] = []
ALL_COLS_NAME: List[str] = []

def set_project_root(project_root: str):
    global PROJECT_ROOT
    PROJECT_ROOT = project_root # type: ignore

def set_output_paths(output_paths: List[str]):
    global OUTPUT_PATHS
    OUTPUT_PATHS = output_paths # type: ignore

def set_config(config: Dict[str, Any]):
    storage_dll_path = config["storage_dll"] 
    storage_dll = os.path.join(PROJECT_ROOT, *storage_dll_path)
    global LIB
    LIB =  ctypes.CDLL(storage_dll)
    LIB.storage_reserve.argtypes = [ctypes.c_size_t]
    LIB.storage_reserve.restype = ctypes.c_void_p
    LIB.storage_commit.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    LIB.storage_commit.restype = None
    
logger = logging.getLogger(__name__)

def storage_data(data_to_flat: Any) -> tuple[int, int]:
    """
    Interfaz para guardar la información generada.
    Retorna una tupla con la dirección de memoria y el tamaño del buffer.
    """
    flat_data = data_to_flat.to_numpy(dtype=str, copy=False).ravel(order="C")
    return _request_storage(flat_data)

def _request_storage(flat_data: Any) -> tuple[int, int]:
    """
    Solicita memoria y escribe bytes en memoria apartada por C++.
    Retorna la dirección de memoria (int) y el tamaño del buffer (int).
    """
    buff_size = sum(len(x.encode("utf-8")) for x in flat_data)
    
    ptr = LIB.storage_reserve(buff_size)
    if not ptr:
        raise MemoryError("La DLL de C++ no pudo reservar la memoria solicitada.")

    offset = 0
    for x in flat_data:
        b = x.encode("utf-8")
        byte_len = len(b)
        ctypes.memmove(ptr + offset, b, byte_len)
        offset += byte_len

    LIB.storage_commit(ptr, buff_size)
    
    return ptr, buff_size