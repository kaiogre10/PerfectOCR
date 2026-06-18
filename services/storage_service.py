from typing import List, Any, Dict
import logging
import os
import ctypes
from services.system_service import get_so

OUTPUT_PATHS: List[str] = []

def storage_config(PROJECT_ROOT: str, config: Dict[str, Any]):
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
        ctypes.POINTER(ctypes.c_size_t), # Arreglo de tamaños individuales de cada string (AHORA EN BYTES UTF-16)
        ctypes.c_size_t                 # Cantidad total de elementos (N)
    ]
    LIB.storage_batch_flat.restype = None

logger = logging.getLogger(__name__)

def storage_data(data_to_flat: Any) -> bool:
    """Interfaz pública para guardar la información generada."""
    # 1. NOTA: Invertimos el orden del return de transform_data para que coincida con tu firma
    buffer_sizes, plain_text = transform_data(data_to_flat)
    return request_storage(plain_text, buffer_sizes)


def request_storage(plain_text: str, buff_sizes: List[int]) -> bool:
    # CAMBIO 1: El tamaño total en bytes es la suma de los tamaños UTF-16 ya calculados
    total_bytes = sum(buff_sizes)
    cantidad_elementos = len(buff_sizes) # N es la cantidad de strings, no el total de bytes

    arreglo_tamanos_c = (ctypes.c_size_t * cantidad_elementos)(*buff_sizes)

    # CAMBIO 2: Convertimos la cadena completa a UTF-16LE de forma directa.
    # .encode("utf-16le") genera exactamente 'total_bytes' de longitud.
    bytes_utf16 = plain_text.encode("utf-16le")
    plaintext_c = (ctypes.c_uint8 * total_bytes).from_buffer_copy(bytes_utf16)

    try:
        # CAMBIO 3: Pasamos la cantidad real de elementos (N) en el tercer argumento
        LIB.storage_batch_flat(plaintext_c, arreglo_tamanos_c, ctypes.c_size_t(cantidad_elementos))
    except ctypes.ArgumentError as e:
        logger.warning(f"Error escribiendo bytecode en memoria asignada por C++: {e}", exc_info=True)
        return False
    return True

def transform_data(df: Any):
    plain_df: List[str] = []
    buffer_sizes: List[int] = []

    for fila in df.itertuples(index=False, name=None):
        fila = list(fila)
        string_row = "".join(fila)[:-1]

        # CAMBIO 4: Medir de forma ultra eficiente el tamaño en bytes de la fila en UTF-16LE.
        # Al multiplicar por 2 evitamos codificar celda por celda en el bucle.
        buff_size_bytes = len(string_row) * 2

        plain_df.append(string_row)
        buffer_sizes.append(buff_size_bytes)

    plain_text = "".join(plain_df)
    return buffer_sizes, plain_text