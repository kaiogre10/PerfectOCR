# core/utils/output_service.py
import os
# from functools import wraps
import logging
import numpy as np
import cv2
import pandas as pd # type: ignore
from typing import Dict, Any, List, Tuple, Optional
from services.log_service import get_caller_info
from utils.file_handler import save_image, save_yaml, save_table
# from collections.abc import Callable
# from typing import TypeAlias

logger = logging.getLogger(__name__)

PROJECT_ROOT: str 
OUTPUT_PATHS: List[str]
TEMP_FILE: str
file_path: str

# SerializerFn: TypeAlias = Callable[[Any, str, str], None]

# _REGISTRY: Dict[str, SerializerFn] = {}

def set_output_config(project_root: str, config: Dict[str, Any]):
    global PROJECT_ROOT, OUTPUT_PATHS, TEMP_FILE, file_path
    PROJECT_ROOT = project_root # type: ignore
    OUTPUT_PATHS = config["output_paths"] # type: ignore
    file_path = os.path.join(PROJECT_ROOT, "core", "assets", "data.npy")
    TEMP_FILE = config.get("temp_path", "") # type: ignore

# def serilizable(file_name: str) -> Callable[[SerializerFn],SerializerFn]:
#     def decorator(fn: SerializerFn) -> SerializerFn:
#         _REGISTRY = fn
#         @wraps(fn)
#         def wrapper(data: Any, output_dir: str, file_name: str) -> None:
#             return fn(data, output_dir, file_name)
#         return wrapper
#     return decorator

# def save_files(items: List[Tuple[Any, str, str]]) -> bool:
#     try:
#         for data, output_dir, file_name in items:
#             _REGISTRY(data, output_dir, file_name[:-4])
#         return True
#     except KeyError as e:
#         logger.warning(f"ERROR GUARDANDO OUTPUTS: {e}", exc_info=True)
#         return False

def save_shapes(image_name: str, poly_id: str, image: np.ndarray[Any, Any], contours1: List[np.ndarray[Any, Any]], contours2: List[np.ndarray[Any, Any]]):
    """Guarda una imagen con los contornos marcados sobre ella"""
    try:
        for _, path in enumerate(OUTPUT_PATHS):
            output_dir = os.path.join(path, image_name)
            file_name = f"{poly_id}.png"            # Dibuja todos los contornos sobre la imagen
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)      # type: ignore
            if contours1 and contours2:
                logger.info("Todos los contornos, contornos 1: Rojo, Contornos 2: Azul")
                cv2.drawContours(image, [np.array(cont, dtype=np.int32) for cont in contours2], -1, (255 ,0, 0), thickness=cv2.FILLED) # AZUL
                cv2.drawContours(image, [np.array(cont, dtype=np.int32) for cont in contours1], -1, (0, 69, 240), thickness=cv2.FILLED) # Rojo
                save_image(image, output_dir, file_name)

            elif not contours1:
                logger.info(f" {len(contours2)} contornos principales 2")
                cv2.drawContours(image, [np.array(cont, dtype=np.int32) for cont in contours2], -1, (0, 69, 240), thickness=-1) # rojo
                save_image(image, output_dir, file_name)
            
            elif not contours2:
                logger.info(f" {len(contours1)} contornos principales 1")
                cv2.drawContours(image, [np.array(cont, dtype=np.int32) for cont in contours1], -1, (0, 69, 240), thickness=cv2.FILLED) # rojo
                save_image(image, output_dir, file_name)
            else:
                logger.error("Se entregaron contornos vacíos")

    except Exception as e:
        logger.error(f"Error guardando contornos: {e}", exc_info=True)

def save_croped_image(image_name: str, img_id: str, image: np.ndarray[Any, Any]): 
    """Guarda una imagen de depuración si la salida está habilitada."""
    worker_name = get_caller_info()[0]
    for path in OUTPUT_PATHS:
        output_dir = os.path.join(path, image_name)
        file_name = f"{img_id}.png"
        
        save_image(image, output_dir, file_name)
        output_dir = os.path.join(path, worker_name, image_name)

    logger.debug(f"Imagenes debug de {worker_name} guardadas")
        
def save_text_debug(results: Dict[str, Any], file_name: str) -> bool:
    worker_name = get_caller_info()[0]
    try:
        results_ser = to_serializable(results)
        for path in OUTPUT_PATHS:
            output_dir = os.path.join(path, worker_name)
            file_name = f"{file_name}_{worker_name}.yaml"
            if save_yaml(results_ser, output_dir, file_name):
                logger.info(f"YAML de {worker_name} generado para '{file_name}'")
                return True

    except Exception as e:
        logger.warning(f"Error guardando {worker_name}.YAML: {e}", exc_info=True)
    return False

def save_table_values(file_name: str, all_features: Dict[str, Dict[str, float]] | np.ndarray[Any, Any]):
    worker_name = get_caller_info()[0]
    try:
        if isinstance(all_features, dict):
            df: pd.DataFrame = pd.DataFrame.from_dict(all_features, orient='index') # type: ignore
            df.index.name = 'line_id'
            df = df.reset_index()
        else:
            
            df: pd.DataFrame = pd.DataFrame(all_features[1:, :])

        for path in OUTPUT_PATHS:
            output_dir = os.path.join(path, worker_name)
            table_file_name = f"{file_name}_{worker_name}.csv"
            save_table(df, output_dir, table_file_name, False)

    except Exception as e:
        logger.error(f"Error calculando Features output: {e}", exc_info=True)

def save_debug_table(corrected_df: pd.DataFrame, file_name: str, output: Optional[bool], stac: Optional[bool]):
    worker_name = get_caller_info()[0]
    try:
        for path in OUTPUT_PATHS:
            output_dir = os.path.join(path, worker_name)
            file_name = f"{file_name}_{worker_name}.csv"
            save_table(corrected_df, output_dir, file_name, (stac if stac else False))

    except Exception as e:
        logger.error(f"Error guardadndo tabla CSV de {worker_name}: {e}", exc_info=True)
        
def to_serializable(obj: Any) -> Any:
    """Convierte numpy arrays y tipos numpy a tipos nativos Python."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()} # type: ignore
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(item) for item in obj] # type: ignore
    else:
        return obj
        
def write_temp_log(payload_temp: Tuple[str, str]) -> bool:
    if not payload_temp[0] or not payload_temp[1]:
        raise TypeError("DATOS PARA REGISTRO VACIOS")
    
    with open(TEMP_FILE, "a", encoding="utf-16-le") as file_temp:
        # time = get_time_stamp(False)
        file_temp.write(f"{payload_temp[1]}\n")
        return True

def serialize_arrays(array_input: np.ndarray[Any, Any]):
    logger.info(f"ARRAY INPUT:\n"f"{array_input.shape}")
    if os.path.exists(file_path):
        data_matrix = np.load(file_path, 'r+', allow_pickle=False)
        ngrams_array = np.ascontiguousarray(np.concatenate((data_matrix, array_input), axis=0, dtype=np.float32), dtype=np.float32)
        logger.info(f"SHAPE: {ngrams_array.shape}")
    else:
        ngrams_array = np.ascontiguousarray(array_input, dtype=np.float32)
    np.save(file_path, ngrams_array, allow_pickle=False)