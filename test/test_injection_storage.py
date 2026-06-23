import os
import sys
import platform
import ctypes
import logging

# Configuración de registro
logging.basicConfig(level='DEBUG', format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Definición estática del payload extraído de su archivo .txt
RAW_PAYLOAD = "3_VINCI 25 AM LIM 50 C/10_85.50_256.50_5_PUNTILLAS PM 0.5_9.49_47.45_1_PASTA CTA SURT C/25_135.00_135.00_2_FOLDER CTA CR C/10_22.50_45.00_1_SOBRE C/ROND CTA C/10_49.80_49.80_2_CREPE NGO C/10_48.00_96.00_1_KM MED NG C/12_39.90_39.90_1_KM FINO NG C/12_42.30_42.30_3_CARTA DIAM CANARIO C/10_30.00_90.00_10_PLACA UNIC 25X25X1_2.98_29.76_1_PALO RED DELG 4X60 K 95_59.28_59.28_1_GLOBO 5 SURT C/100_34.93_34.93_2_CART FLUORESC AM C/10_49.10_98.20_"

def test_load_native_library(project_root: str) -> ctypes.CDLL:
    """Resuelve la ruta y carga la librería dinámica correspondiente al OS."""
    if platform.system() == "Windows":
        binary_extension = ".dll"
    elif platform.system() == "Linux":
        binary_extension = ".so"
    else:
        binary_extension = ".dylib"

    storage_bin_path = os.path.join(project_root, "core", "utils", "compiled_utils", "buffer_handler" + binary_extension)
    
    if not os.path.exists(storage_bin_path):
        logger.error(f"BINARIO NO ENCONTRADO: {storage_bin_path}")
        sys.exit(1)

    lib = ctypes.CDLL(storage_bin_path)
    
    # Configuración de la firma de la función nativa
    lib.storage_batch_flat.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),  
        ctypes.POINTER(ctypes.c_size_t), 
        ctypes.c_size_t                  
    ]
    lib.storage_batch_flat.restype = None
    return lib

def test_execute_injection_test(lib: ctypes.CDLL):
    """Procesa el payload y lo inyecta en la memoria del binario C++."""
    # 1. Extracción y validación de elementos (13 filas * 4 columnas = 52)
    elementos = [elem.strip() for elem in RAW_PAYLOAD.split("_") if elem.strip()]
    total_elements = len(elementos)
    
    if total_elements != 52:
        logger.warning(f"Inconsistencia de estructura: Se esperaban 52 elementos, detectados {total_elements}")

    # 2. Transformación y cálculo de offsets para C++
    buff_sizes = []
    plain_text_str = ""
    
    for cell in elementos:
        # Requisito de arquitectura: Duplicar bytes para codificación UTF-16LE
        buff_size_bytes = len(cell) * 2
        buff_sizes.append(buff_size_bytes)
        plain_text_str += cell

    total_bytes = sum(buff_sizes)

    # 3. Serialización del texto plano
    try:
        bytes_utf16 = plain_text_str.encode("utf-16le")
    except Exception as e:
        logger.error(f"Falla de codificación: {e}")
        return

    # Verificación de integridad de memoria
    if len(bytes_utf16) != total_bytes:
        logger.error(f"Desfase de memoria crítico: {total_bytes} bytes calculados frente a {len(bytes_utf16)} reales.")
        return

    # 4. Asignación de estructuras Ctypes
    arreglo_tamanos_c = (ctypes.c_size_t * total_elements)(*buff_sizes)
    plaintext_c = (ctypes.c_uint8 * total_bytes).from_buffer_copy(bytes_utf16)

    # 5. Inyección de memoria (Invocación a C++)
    logger.info("EJECUTANDO INYECCIÓN A LIBRERÍA NATIVA...")
    logger.info(f"-> Elementos inyectados: {total_elements}")
    logger.info(f"-> Bytes mapeados: {total_bytes}")
    
    try:
        lib.storage_batch_flat(plaintext_c, arreglo_tamanos_c, ctypes.c_size_t(total_elements))
        logger.info("INYECCIÓN COMPLETADA CON ÉXITO.")
    except Exception as e:
        logger.error(f"ERROR DURANTE LA EJECUCIÓN DEL BINARIO: {e}", exc_info=True)

if __name__ == "__main__":
    # Determinación estricta de la ruta del proyecto
    PROJECT_ROOT = "C:\\PerfectOCR"
    
    logger.info(f"Directorio de trabajo: {PROJECT_ROOT}")
    
    native_lib = test_load_native_library(PROJECT_ROOT)
    test_execute_injection_test(native_lib)