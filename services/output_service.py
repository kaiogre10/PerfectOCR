# core/utils/output_service.py
import os
import json
import logging
import numpy as np
import cv2
import pandas as pd # type: ignore
from typing import Dict, Any, List
import csv
from core.utils.data_utils import FEATURES_NAME

OUTPUT_PATHS: List[str] = []

def set_output_paths(output_paths: List[str]):
    global OUTPUT_PATHS
    OUTPUT_PATHS = output_paths # type: ignore

logger = logging.getLogger(__name__)

def save_shapes(image_name: str, poly_id: str, image: np.ndarray[Any, Any], contours1: List[np.ndarray[Any, Any]], contours2: List[np.ndarray[Any, Any]]):
    """Guarda una imagen con los contornos marcados sobre ella"""
    try:
        for path in OUTPUT_PATHS:
            output_dir = path
            file_name = f"{image_name}_{poly_id}.png"            # Dibuja todos los contornos sobre la imagen
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)      # type: ignore
            if contours1 and contours2:
                logger.info("Todos los contornos, contornos 1: Rojo, Contornos 2: Azul")
                cv2.drawContours(image, [np.array(cont, dtype=np.int32) for cont in contours2], -1, (255 ,0, 0), thickness=cv2.FILLED) # AZUL
                cv2.drawContours(image, [np.array(cont, dtype=np.int32) for cont in contours1], -1, (0, 69, 240), thickness=cv2.FILLED) # Rojo
                save_image(image, output_dir, file_name)

            elif not contours1:
                logger.info(f" {len(contours2)} contornos principales 2")
                cv2.drawContours(image, [np.array(cont, dtype=np.int32) for cont in contours2], -1, (0, 69, 240), thickness=cv2.FILLED) # rojo
                save_image(image, output_dir, file_name)
            
            elif not contours2:
                logger.info(f" {len(contours1)} contornos principales 1")
                cv2.drawContours(image, [np.array(cont, dtype=np.int32) for cont in contours1], -1, (0, 69, 240), thickness=cv2.FILLED) # rojo
                save_image(image, output_dir, file_name)
            else:
                logger.error("Se entregaron contornos vacíos")

    except Exception as e:
        logger.error(f"Error guardando contornos: {e}", exc_info=True)

def save_croped_image(image_name: str, img_id: str, image: np.ndarray[Any, Any], worker_name: str): 
    """Guarda una imagen de depuración si la salida está habilitada."""
    for path in OUTPUT_PATHS:
        output_dir = os.path.join(path, image_name)
        file_name = f"{img_id}.png"
        save_image(image, output_dir, file_name)
        output_dir = os.path.join(path, worker_name, image_name)

    logger.debug(f"Imagenes debug de {worker_name} guardadas")

def save_image(image: np.ndarray[Any, np.dtype[np.uint8]], output_dir: str, file_name: str):
    """Guarda una única imagen en disco."""
    try:    
        os.makedirs(output_dir, exist_ok=True)
        img_path = os.path.join(output_dir, file_name)
        cv2.imwrite(img_path, image)
        
        return img_path
    except Exception as e:
        logger.error(f"Error guardando '{file_name}' imagen: {e}")
        
def save_debug_json(worker_name: str, results: Dict[str, Any], file_name: str):
    try:
        final_results: Dict[str, Any] = {}
        for line_id in results:
            if line_id in results:
                line_obj = results[line_id]
                final_results[line_id] = {
                    'lineal_id': line_obj.lineal_id,
                    'text': line_obj.text,
                    'polygon_ids': line_obj.polygon_ids,
                }
                
        for path in OUTPUT_PATHS:
            output_dir = os.path.join(path, worker_name)
            file_name = f"{file_name}_{worker_name}.json"
            save_json(final_results, output_dir, file_name)

        logger.warning(f"JSON de {worker_name} generado para '{file_name}'.")

    except Exception as e:
        logger.warning(f"Error guardando {worker_name}.JSON: {e}", exc_info=True)
    
def save_raw_json(worker_name: str, results: Dict[str, Any], file_name: str) -> bool:
    try:
        results_ser = to_serializable(results)
        for path in OUTPUT_PATHS:
            output_dir = os.path.join(path, worker_name)
            file_name = f"{file_name}_{worker_name}.json"
            if save_json(results_ser, output_dir, file_name):
                logger.warning(f"JSON de {worker_name} generado para '{file_name}'.")
                return True
        
        return False
        
    except Exception as e:
        logger.warning(f"Error guardando {worker_name}.JSON: {e}", exc_info=True)
        return False

def save_json(results: Dict[str, Dict[str, Any]], output_dir: str, file_name: str) -> bool:
    """Guarda un JSON en disco."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, file_name)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
            
        return True
    except Exception as e:
        logger.error(f"Error guardando JSON: {e}", exc_info=True)
        return False
        
def save_debug_table(corrected_df: pd.DataFrame, file_name: str, worker_name: str, header_polygons: List[Any]):
    try:
        header_text: List[str] = []
        for poly_obj in header_polygons:
            poly_text = getattr(poly_obj, "ocr_text", None)
            if poly_text:
                header_text.append(poly_text)
        if not header_text:
            header_text = list(corrected_df.columns)

        for path in OUTPUT_PATHS:
            output_dir = os.path.join(path, worker_name)
            file_name = f"{file_name}_{worker_name}.csv"
            save_table(corrected_df, output_dir, file_name, header_text)

    except Exception as e:
        logger.error(f"Error guardadndo tabla JSON de {worker_name},: {e}", exc_info=True)

def save_table_values(file_name: str, all_features: Dict[str, Dict[str, float]] | np.ndarray[Any, Any], worker_name: str):
    feature_names = FEATURES_NAME
    try:
        if isinstance(all_features, dict):
            df: pd.DataFrame = pd.DataFrame.from_dict(all_features, orient='index') # type: ignore
            df.index.name = 'line_id'
            df = df.reset_index()
            header = list(df.columns)
        else:
            
            df: pd.DataFrame = pd.DataFrame(all_features[1:, :])
            header = list(feature_names)

        for path in OUTPUT_PATHS:
            output_dir = os.path.join(path, worker_name)
            table_file_name = f"{file_name}_{worker_name}.csv"
            save_table(df, output_dir, table_file_name, header)

    except Exception as e:
        logger.error(f"Error calculando Features output: {e}", exc_info=True)
        
def save_table(corrected_df: pd.DataFrame, output_dir: str, file_name: str, header_text: List[str]):
    """
    Guarda una tabla estructurada en formato CSV (compatible con Excel).
    Ruta del archivo guardado o None si hay error.
    """
    try:      
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, file_name)
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header_text)
            # Escribimos las filas del DataFrame, no solo los nombres de columnas
            for row in corrected_df.itertuples(index=False, name=None):
                writer.writerow(row)                
        try:
            _append_table_to_master(
                corrected_df=corrected_df,
                output_dir=output_dir,
                # section_title=os.path.splitext(os.path.basename(file_name))[0],
                header_text=header_text,
                master_filename="tables_master.csv"
            )
        except Exception as e:
            logger.error(f"error generando el tables_master: {e}", exc_info=True)
        
        logger.info(f"Tabla debug generada de: {file_name}")
                        
        return output_file
    except Exception as e:
        logger.error(f"Error guardando CSV: {e}", exc_info=True)
        
def _append_table_to_master(corrected_df: pd.DataFrame, output_dir: str, header_text: List[str], master_filename: str = "tables_master.csv"):
    os.makedirs(output_dir, exist_ok=True)
    master_path = os.path.join(output_dir, master_filename)
    write_header = not os.path.exists(master_path) or os.path.getsize(master_path) == 0

    with open(master_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header_text if (header_text and len(header_text) > 0) else list(corrected_df.columns))
        for row in corrected_df.itertuples(index=False, name=None):
            writer.writerow(row)

def to_serializable(obj: Any) -> Any:
    """Convierte numpy arrays y tipos numpy a tipos nativos Python."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()} #type: ignore
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(item) for item in obj]#type: ignore
    else:
        return obj
