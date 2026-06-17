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

    container_bin_path = config["container_bin"]
    container_ext = container_bin_path.pop(-1)
    container_extension = (container_ext + binary_extension)
    container_bin = os.path.join(PROJECT_ROOT, *container_bin_path, container_extension)
    try:
        CON = ctypes.CDLL(container_bin)
        CON.container_create.argtypes = [ctypes.c_int]
        CON.container_create.restype = None
        CON.container_create(1)
    except BaseExceptionGroup as e:
        logger.error(f"NO SE PUDO PUDO INICIAR EL CONTENDOR EN MEMORIA: {e}", exc_info=True)
    
    storage_bin_path = config["storage_bin"]
    binary_ext = storage_bin_path.pop(-1)
    binary_extension = (binary_ext + binary_extension)
    storage_bin = os.path.join(PROJECT_ROOT, *storage_bin_path, binary_extension)

    global LIB # type: ignore
    try:
        LIB = ctypes.CDLL(storage_bin)
    except FileNotFoundError as e:
        logger.warning(f"ERROR CARGANDO EL BINARIO: {e}", exc_info=True)
        return None

    LIB.storage_batch_flat.argtypes = [
    ctypes.POINTER(ctypes.c_uint8),  # Puntero a los bytes contiguos aplanados
    ctypes.POINTER(ctypes.c_size_t), # Arreglo de tamaños individuales de cada string
    ctypes.c_size_t                 # Cantidad total de elementos (N)
    ]
    LIB.storage_batch_flat.restype = None

logger = logging.getLogger(__name__)

def storage_data(data_to_flat: Any) -> Tuple[int, List[int]]:
    """
    Interfaz pública para guardar la información generada.
    Retorna una tupla con la dirección de memoria y el tamaño del buffer.
    """
    # flat_data = data_to_flat.to_numpy(dtype=str, copy=False).ravel(order="C")
    # buff_size = sum(len(x.encode("utf-8")) for x in flat_data)
    buff_size, plain_text = transform_data(data_to_flat)
    #logger.info(f"ANTES DE INGRESAR: Flat data:'{flat_data}'\n"f"TAMAÑO BYTES NP.ARRAY: {flat_data.nbytes}'B\n"f"TAMAÑO DF: {data_to_flat.memory_usage(index=True, deep=True).sum()}'B\n"f"TAMAÑO BYTES MEMORIA: '{buff_size}'B'")
    ptr, buff_size = _request_storage(plain_text, buff_size)
    return ptr, buff_size

def _request_storage(plain_text: str , buff_sizes: List[int]) -> Tuple[int, List[int]]:
    byte_len = sum(buff_sizes)
    arreglo_tamanos_c = (ctypes.c_size_t * byte_len)(*buff_sizes)
    plaintext = (ctypes.c_uint8 * byte_len)(*plain_text.encode("utf-8"))
    try:
        # logger.info(f"PUNT: '{plaintext}', | memory view: {bytes(plaintext)}")
        LIB.storage_batch_flat(plaintext, arreglo_tamanos_c, ctypes.c_size_t(byte_len))
    except ctypes.ArgumentError as e:
        logger.warning(f"Error escribiendo bytecode en memoria asignada por C++: {e}", exc_info=True)
        return (0, [])
    return (0, [])

def transform_data(df: Any):
    plain_df: List[str] = []
    buffer_sizes: List[int] = []
    plain_text: str = ""
    for fila in df.itertuples(index=False, name=None):
        fila = list(fila)
        string_row = "".join(fila)[:-1]
        buff_size = sum(len(x.encode("utf-8")) for x in string_row)
        plain_df.append(string_row)
        buffer_sizes.append(buff_size)

    plain_text = "".join(plain_df)
    # logger.info(f"BUUF_SIZES: {buffer_sizes}\n"f"DF PLANO: {plain_df}\n"f"TEXTO PLANO: '{plain_text}'")
    return buffer_sizes, plain_text

# Para acceder desde cualquier lado
#base_ptr, offsets = _request_storage(plano, BUUF_SIZES)
# elemento i → base_ptr + offsets[i], longitud → offsets[i+1] - offsets[i]