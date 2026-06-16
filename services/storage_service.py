from typing import List, Any, Dict, Tuple
import logging
import os
import ctypes
from services.system_service import get_so

PROJECT_ROOT: str = ""
OUTPUT_PATHS: List[str] = []

def set_project_root(project_root: str):
    global PROJECT_ROOT
    PROJECT_ROOT = project_root # type: ignore

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
        
    LIB.storage_reserve.argtypes = [
        ctypes.POINTER(ctypes.c_char_p),   # strings
        ctypes.POINTER(ctypes.c_size_t),   # sizes
        ctypes.c_size_t,                   # count
        ctypes.POINTER(ctypes.c_size_t),   # offsets_out
    ]
    LIB.storage_reserve.restype = ctypes.c_void_p

    LIB.storage_free.argtypes = [ctypes.c_void_p]
    LIB.storage_free.restype  = None
    
logger = logging.getLogger(__name__)

def storage_data(data_to_flat: Any) -> Tuple[int, List[int]]:
    """
    Interfaz pública para guardar la información generada.
    Retorna una tupla con la dirección de memoria y el tamaño del buffer.
    """
    # flat_data = data_to_flat.to_numpy(dtype=str, copy=False).ravel(order="C")
    # buff_size = sum(len(x.encode("utf-8")) for x in flat_data)
    flat_data, buff_size = transform_data(data_to_flat)
    #logger.info(f"ANTES DE INGRESAR: Flat data:'{flat_data}'\n"f"TAMAÑO BYTES NP.ARRAY: {flat_data.nbytes}'B\n"f"TAMAÑO DF: {data_to_flat.memory_usage(index=True, deep=True).sum()}'B\n"f"TAMAÑO BYTES MEMORIA: '{buff_size}'B'")
    ptr, buff_size = _request_storage(flat_data, buff_size)
    return ptr, buff_size

def _request_storage(flat_data: List[str], buff_sizes: List[int]) -> Tuple[int, List[int]]:
    try:
        count = len(flat_data)
        try:
            c_strings  = (ctypes.c_char_p * count)(*[s.encode("utf-8") for s in flat_data])
            c_sizes    = (ctypes.c_size_t * count)(*buff_sizes)
            c_offsets  = (ctypes.c_size_t * (count + 1))()  # +1 sentinel
        except TypeError as e:
            logger.info(f"ERROR DE TYPO: {e}", exc_info=True)
            return (0, [])

        try:
            LIB.storage_reserve.restype  = ctypes.c_void_p
            LIB.storage_reserve.argtypes = [
                ctypes.POINTER(ctypes.c_char_p),
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t),
            ]
        except MemoryError as e:
            logger.error(f"ERROR EN MEMORIA: {e}", exc_info=True)
            return (0, [])

        try:
            ptr = LIB.storage_reserve(c_strings, c_sizes, count, c_offsets)
            if not ptr:
                logger.info(f"ptr raw: {ptr}, offsets: {list(c_offsets)}")
                raise MemoryError
        except OSError as e:
            logger.info(f"ERROR EN MEMORUA: {e}", exc_info=True)
            return (0, [])

        offsets = list(c_offsets)  # count+1 valores, último es total
        logger.info(f"PTR: {ptr}, HEX: '{hex(ptr)}', {offsets}")
        return (ptr, offsets)
    
    except Exception as e:
        logger.error(f"Error conectadno con C++: {e}", exc_info=True)
    return (0, [])

def transform_data(df: Any) -> Tuple[List[str], List[int]]:
    plain_df: List[str] = []
    buffer_sizes: List[int] = []
    for fila in df.itertuples(index=False, name=None):
        fila = list(fila)
        string_row = "".join(fila)[:-1]
        buff_size = sum(len(x.encode("utf-8")) for x in string_row)
        plain_df.append(string_row)
        buffer_sizes.append(buff_size)

    logger.info(f"BUUF_SIZES: {buffer_sizes}\n"f"PLANO: {plain_df}")

    return plain_df, buffer_sizes

# Para acceder desde cualquier lado
#base_ptr, offsets = _request_storage(plano, BUUF_SIZES)
# elemento i → base_ptr + offsets[i], longitud → offsets[i+1] - offsets[i]