from typing import List, Dict
import logging
import os
import ctypes
from services.system_service import get_so

OUTPUT_PATHS: List[str] = []
CON: ctypes.CDLL
LIB: ctypes.CDLL

logger = logging.getLogger(__name__)

def storage_config(PROJECT_ROOT: str, config: Dict[str, List[str]]) -> None:
    os_bins_ext: str = get_so()
    bin_path = config["libs_path"]
    bins_path = [PROJECT_ROOT, *bin_path]

    try:
        container_bin = os.path.join(*bins_path, f"{"containers" + os_bins_ext}")
        CON = ctypes.CDLL(container_bin) # type: ignore
        CON.container_create.argtypes = [ctypes.c_int]
        CON.container_create.restype = None
        CON.container_create(1)
    except BaseExceptionGroup as e:
        logger.error(f"NO SE PUDO PUDO INICIAR EL CONTENDOR EN MEMORIA: {e}", exc_info=True)
        raise

    global LIB # type: ignore
    try:
        storage_bin = os.path.join(*bins_path, f"{"buffer_handler" + os_bins_ext}")
        LIB = ctypes.CDLL(storage_bin) # type: ignore
    except FileNotFoundError as e:
        logger.warning(f"ERROR CARGANDO BUFFER HANDLER: {e}", exc_info=True)
        raise

    LIB.storage_batch_flat.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),  # Puntero a los bytes contiguos aplanados
        ctypes.POINTER(ctypes.c_size_t), # Arreglo de tamaños individuales de cada string (AHORA EN BYTES UTF-16)
        ctypes.c_size_t                 # Cantidad total de elementos (N)
    ]
    LIB.storage_batch_flat.restype = None

def storage_data(plain_text: str, buffer_sizes: List[int]) -> bool:
    """Interfaz pública para guardar la información generada."""
    return _request_storage(plain_text, buffer_sizes)

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
            #time0 = time.perf_counter()
            LIB.storage_batch_flat(plaintext_c, arreglo_tamanos_c, ctypes.c_size_t(total_rows))
            #logger.info(f"TIEMPO ESCRIBIENDO '{sum(buff_sizes)}' Bytes EN MEMORIA: {time.perf_counter() - time0:.8f}'s")
        except ctypes.ArgumentError as e:
            logger.warning(f"Error escribiendo bytecode en memoria asignada por C++: {e}", exc_info=True)
            raise
        return True
    except Exception as e:
        logger.error(f"ERROR SOLICITANDO MEMORIA: {e}", exc_info=True)
    return False