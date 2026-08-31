
from typing import List, Dict, Optional
import logging
import ctypes

LIB: ctypes.CDLL

logger = logging.getLogger(__name__)

def storage_config(config: Dict[str, List[str]]) -> None:
    container_bin_path = config.get("containers", "")
    storage_bin_path = config.get("buffer_handler", "")
    try:
        CON = ctypes.CDLL(container_bin_path)
        if not CON:
            raise OSError("ERROR CARGANDO CONTENEDOR")
        
        CON.container_create.argtypes = [ctypes.c_int]
        CON.container_create.restype = None
        CON.container_create(1)
    except BaseExceptionGroup as e:
        logger.error(f"NO SE PUDO PUDO INICIAR EL CONTENDOR EN MEMORIA: {e}", exc_info=True)
        raise

    global LIB # type: ignore
    try:
        LIB = ctypes.CDLL(storage_bin_path) # type: ignore
        if not LIB:
            raise OSError("ERROR CARGANDO BUFFER")
        
        LIB.reserve_buffer.argtypes = [ctypes.c_size_t]
        LIB.reserve_buffer.restype = ctypes.c_void_p
        LIB.commit_buffer.argtypes = [ctypes.c_int]
        LIB.commit_buffer.restype = None
    except FileNotFoundError as e:
        logger.warning(f"ERROR CARGANDO BUFFER HANDLER: {e}", exc_info=True)
        raise

def storage_data(texts: List[str]) -> Optional[List[int]]:
    buffers: List[int] = []
    for plain_texts in texts:
        raw_payload = plain_texts.encode("ascii", "ignore")
        if not raw_payload:
            raise EncodingWarning("ERROR CONVIRTIENDO BYTES")
        
        len_bytes = len(raw_payload) * 2
        
        if len_bytes < len(plain_texts):
            raise ArithmeticError("ERROR TAMAÑO BYTES")

        ptr = LIB.reserve_buffer(len_bytes) # Solicitar memoria
        if not ptr:
            raise ctypes.ArgumentError("ERROR DE PUNTEROS")
            
        ctypes.memmove(ptr, raw_payload.decode("ascii").encode("utf-16-le"), len_bytes) # Guardar bytes typados
        LIB.commit_buffer(1) # Avisar que ya están guardados
        buffers.append(len_bytes)

    return buffers