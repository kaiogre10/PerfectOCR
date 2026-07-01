from typing import List, Dict, Optional
import logging
import os
import ctypes
from services.system_service import get_so

LIB: ctypes.CDLL

logger = logging.getLogger(__name__)

def storage_config(PROJECT_ROOT: str, config: Dict[str, List[str]]) -> None:
    bin_path = config["libs_path"]
    bins_path = [PROJECT_ROOT, *bin_path]
    binary_extension: str = get_so()

    container_bin_path = os.path.join(*bins_path, "containers" + binary_extension)
    storage_bin_path = os.path.join(*bins_path, "buffer_handler" + binary_extension)
    if not os.path.exists(storage_bin_path) or not os.path.exists(container_bin_path):
            logger.error(f"BINARIOS NO ENCONTRADO: {storage_bin_path}")
    try:
        CON = ctypes.CDLL(container_bin_path)
        CON.container_create.argtypes = [ctypes.c_int]
        CON.container_create.restype = None
        CON.container_create(1)
    except BaseExceptionGroup as e:
        logger.error(f"NO SE PUDO PUDO INICIAR EL CONTENDOR EN MEMORIA: {e}", exc_info=True)
        raise

    global LIB # type: ignore
    try:
        LIB = ctypes.CDLL(storage_bin_path) # type: ignore
        LIB.reserve_buffer.argtypes = [ctypes.c_size_t]
        LIB.reserve_buffer.restype = ctypes.c_void_p
        LIB.commit_buffer.argtypes = [ctypes.c_int]
        LIB.commit_buffer.restype = None
    except FileNotFoundError as e:
        logger.warning(f"ERROR CARGANDO BUFFER HANDLER: {e}", exc_info=True)
        raise

def storage_data(plain_text: str) -> Optional[int]:
    try:
        raw_payload = plain_text.encode("ascii", errors="ignore")
        len_bytes = (len(raw_payload) * 2)

        ptr = LIB.reserve_buffer(len_bytes) # Solicitar memoria

        ctypes.memmove(ptr, raw_payload.decode("ascii").encode("utf-16-le"), len_bytes) # Guardar bytes typados
        LIB.commit_buffer(1) # Avisar que ya están guardados

        return len_bytes
    
    except ctypes.ArgumentError as e:
        logger.error(f"ERROR GUARDANDO EN MEMORIA: {e}", exc_info=True)
    return None
