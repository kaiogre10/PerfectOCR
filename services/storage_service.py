from typing import List, Any, Dict, Tuple
import logging
import os
import time
import ctypes
from services.system_service import get_so
from services.log_service import log_simple

OUTPUT_PATHS: List[str] = []
CON: ctypes.CDLL
LIB: ctypes.CDLL

logger = logging.getLogger(__name__)

def storage_config(PROJECT_ROOT: str, config: Dict[str, List[str]]) -> None:
    log_simple("STORAGE ACTIVADO")
    binary_extension: str = get_so()
    container_bin_path = config["container_bin"]
    container_ext = container_bin_path.pop(-1)
    container_extension = (container_ext + binary_extension)
    container_bin = os.path.join(PROJECT_ROOT, *container_bin_path, container_extension)
    try:
        CON = ctypes.CDLL(container_bin) # type: ignore
        CON.container_create.argtypes = [ctypes.c_int]
        CON.container_create.restype = None
        CON.container_create(1)
    except BaseExceptionGroup as e:
        logger.error(f"NO SE PUDO PUDO INICIAR EL CONTENDOR EN MEMORIA: {e}", exc_info=True)
        raise
    
    storage_bin_path = config["storage_bin"]
    binary_ext = storage_bin_path.pop(-1)
    binary_extension = (binary_ext + binary_extension)
    storage_bin = os.path.join(PROJECT_ROOT, *storage_bin_path, binary_extension)

    global LIB # type: ignore
    try:
        LIB = ctypes.CDLL(storage_bin) # type: ignore
    except FileNotFoundError as e:
        logger.warning(f"ERROR CARGANDO EL BINARIO: {e}", exc_info=True)
        raise

    LIB.storage_batch_flat.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),  # Puntero a los bytes contiguos aplanados
        ctypes.POINTER(ctypes.c_size_t), # Arreglo de tamaños individuales de cada string (AHORA EN BYTES UTF-16)
        ctypes.c_size_t                 # Cantidad total de elementos (N)
    ]
    LIB.storage_batch_flat.restype = None

def storage_data(data_to_flat: Any) -> List[int]:
    """Interfaz pública para guardar la información generada."""
    buffer_sizes, plain_text = transform_data(data_to_flat)
    if _request_storage(plain_text, buffer_sizes):
        return buffer_sizes
    else:
        return []

def _request_storage(plain_text: str, buff_sizes: List[int]) -> bool:
    try:
        # El tamaño total en bytes es la suma de los tamaños UTF-16 ya calculados
        total_bytes = sum(buff_sizes)               
        total_rows = len(buff_sizes)                                                     # Tma de columnas de strings planos, no el total de bytes
        try:
            # Convertir la cadena completa a UTF-16LE de forma directa.
            # .encode("utf-16le") genera exactamente 'total_bytes' de longitud.
            arreglo_tamanos_c = (ctypes.c_size_t * total_rows)(*buff_sizes)             # Tamaño de cada fila aplanada
            bytes_utf16 = plain_text.encode("utf-16le")
            plaintext_c = (ctypes.c_uint8 * total_bytes).from_buffer_copy(bytes_utf16)  # Texto aplanado
            # Pasar la cantidad real de elementos (N) en el tercer argumento
            time0 = time.perf_counter()
            LIB.storage_batch_flat(plaintext_c, arreglo_tamanos_c, ctypes.c_size_t(total_rows))
            logger.info(f"TIEMPO ESCRIBIENDO '{sum(buff_sizes)}' Bytes EN MEMORIA: {time.perf_counter() - time0:.8f}'s")
        except ctypes.ArgumentError as e:
            logger.warning(f"Error escribiendo bytecode en memoria asignada por C++: {e}", exc_info=True)
            raise
        return True
    except Exception as e:
        logger.error(f"ERROR SOLICITANDO MEMORIA: {e}", exc_info=True)
    return False

def transform_data(df: Any) -> Tuple[List[int], str]:
    """Devuelve tamaño de cada fila y el df aplanado"""
    plain_df: List[str] = []
    buffer_sizes: List[int] = []
    total_rows = df.shape[0]

    for i, fila in enumerate(df.itertuples(index=False, name=None)):
        fila = list(fila)
        if (i + 1) != total_rows:
            string_row = "".join(fila)[:-1]
        else:
            string_row = "".join(fila)

        # Al multiplicar por 2 evitamos codificar celda por celda en el bucle.
        buff_size_bytes = len(string_row) * 2

        plain_df.append(string_row)
        buffer_sizes.append(buff_size_bytes)

    plain_text = "".join(plain_df)
    #logger.info(f"TAMAÑO: '{buffer_sizes}' PLAIN TEXT:\n"f"'{plain_text}'")
    return buffer_sizes, plain_text