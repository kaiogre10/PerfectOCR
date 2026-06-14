from typing import List, Any, Dict, Tuple
import logging
import os
import ctypes
from services.system_service import get_so

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
    binary_extension: str = get_so()
    storage_bin_path = config["storage_bin"]

    binary_ext = storage_bin_path.pop(-1)
    binary_extension = (binary_ext + binary_extension)
    storage_bin = os.path.join(PROJECT_ROOT, *storage_bin_path, binary_extension)

    global LIB # type: ignore
    try:
        LIB = ctypes.CDLL(storage_bin)
    except OSError as e:
        logger.warning(f"ERROR CARGANDO EL BINARIO: {e}", exc_info=True)
        return None
        
    LIB.storage_reserve.argtypes = [ctypes.c_size_t]
    LIB.storage_reserve.restype = ctypes.c_void_p
    LIB.storage_commit.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    LIB.storage_commit.restype = None
    LIB.storage_free.argtypes = [ctypes.c_void_p]
    LIB.storage_free.restype = None
    
logger = logging.getLogger(__name__)

def storage_data(data_to_flat: Any) -> Tuple[int, int]:
    """
    Interfaz para guardar la información generada.
    Retorna una tupla con la dirección de memoria y el tamaño del buffer.
    """
    flat_data = data_to_flat.to_numpy(dtype=str, copy=False).ravel(order="C")
    buff_size = sum(len(x.encode("utf-8")) for x in flat_data)
    logger.info(f"Flat data:'{flat_data}'\n"f"TAMAÑO BYTES ARRAY: {flat_data.nbytes}'B\n"f"TAMAÑO DF: {data_to_flat.memory_usage(index=True, deep=True).sum()}'B\n"f"TAMAÑO BYTES MEMORIA: '{buff_size}'B'")
    ptr, buff_size = _request_storage(flat_data, buff_size)
    return ptr, buff_size

def _request_storage(flat_data: Any, buff_size: int) -> Tuple[int, int]:
    """
    Solicita memoria y escribe bytes en memoria apartada por C++.
    Retorna la dirección de memoria (int) y el tamaño del buffer (int).
    """
    try:
        ptr = LIB.storage_reserve(buff_size)
    except MemoryError as e:
        logger.warning(f"La DLL de C++ no pudo reservar la memoria solicitada: {e}", exc_info=True)
        return (0, 0)
    
    try:
        offset = 0
        for x in flat_data:
            b = x.encode("utf-8")
            byte_len = len(b)
            ctypes.memmove(ptr + offset, b, byte_len)
            offset += byte_len

        LIB.storage_commit(ptr, buff_size)
    except BufferError as e:
        logger.warning(f"Error escribiendo bytecode en memoria asignada por C++: {e}", exc_info=True)
        return (0, 0)
    
    # try:
    #     bytes_leidos = ctypes.string_at(ptr, buff_size)
    # except MemoryError as e:
    #     logger.warning(f"Error leyendo bytecode: {e}", exc_info=True)
    #     return (ptr, buff_size)

    # logger.info(f"\n"f"Dirección: '{ptr} | '{hex(ptr)}', TAMAÑO EN BYTES: '{buff_size}'B'")
    # LIB.storage_free(ptr)

    return ptr, buff_size
