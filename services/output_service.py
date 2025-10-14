# core/utils/output_service.py
import os
import json
import cv2
import logging
import csv
import numpy as np
import pandas as pd # type: ignore
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def save_croped_image(poly_id: str, image: np.ndarray[Any, Any], output_paths: List[str] | str, worker_name: str):
    """Guarda una imagen de depuración si la salida está habilitada."""
    if isinstance(output_paths, str):
        output_paths = [output_paths]
    for path in output_paths:
        output_dir = os.path.join(path, worker_name)
        file_name = f"{poly_id}_{worker_name}.png"
        save_image(image, output_dir, file_name)
        
    logger.debug(f"Imagenes debug de {worker_name} guardadas")

def save_image(image: np.ndarray[Any, np.dtype[np.uint8]], output_dir: str, file_name_with_extension: str):
    """Guarda una única imagen en disco."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        img_path = os.path.join(output_dir, file_name_with_extension)
        cv2.imwrite(img_path, image)
        return img_path
    except Exception as e:
        logger.error(f"Error guardando imagen: {e}")
        
def save_debug_json(output_paths: List[str] | str, worker_name: str, results: Dict[str, Any], file_name: str):
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
                
        if isinstance(output_paths, str):
            output_paths = [output_paths]
        for path in output_paths:
            output_dir = os.path.join(path, worker_name)
            file_name = f"{file_name}_{worker_name}.json"
            save_json(final_results, output_dir, file_name)
            
        logger.warning(f"Output JSON geneado para:'{file_name}'.")
        
    except Exception as e:
        logger.error(f"Error guardando output JSON: {e}", exc_info=True)
    
def save_debug_ocr(output_paths: List[str] | str, worker_name: str, results: Dict[str, Any], file_name: str):
    try:
        if isinstance(output_paths, str):
            output_paths = [output_paths]
        for path in output_paths:
            output_dir = os.path.join(path, worker_name)
            file_name = f"{file_name}_{worker_name}.json"
            save_json(results, output_dir, file_name)
            
        logger.warning(f"OCR Raw results para '{file_name}'.")
        
    except Exception as e:
        logger.warning(f"Error guardando output JSON: {e}", exc_info=True)

def save_json(results: Dict[str, Dict[str, Any]], output_dir: str, file_name: str):
    """Guarda un JSON en disco."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, file_name)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        return output_file
    except Exception as e:
        logger.error(f"Error guardando JSON: {e}", exc_info=True)
        return None
        
def save_debug_table(corrected_df: pd.DataFrame, file_name: str, output_paths: List[str] | str, worker_name: str, all_lines: Dict[str, Any]):
    try:
        
        header_text: List[str] = []
        
        for line_obj in all_lines.values():
            poly_ids_line = getattr(line_obj, "polygon_ids", []) or []
            header_line = [lid for lid, l in all_lines.items() if getattr(l, "header_line", not None)]
            if header_line:
                header_text = line_obj.text
                for poly_id in poly_ids_line:
                    if poly_id in poly_ids_line:
                        poly_text = poly_ids_line[poly_id].ocr_text
                        if poly_text:        
                            header_text.append(poly_text)
                break
    
        if not header_text:
            header_text = list(corrected_df.columns)
                    
        for path in output_paths:
            output_dir = os.path.join(path, worker_name)
            file_name = f"{file_name}_{worker_name}.csv"
            save_table(corrected_df, output_dir, file_name, header_text)
        
    except Exception as e:
        logger.error(f"Error guardadndo tabla JSON de {worker_name},: {e}", exc_info=True)
        
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
                section_title=os.path.splitext(os.path.basename(file_name))[0],
                header_text=header_text,
                master_filename="tables_master.csv"
            )
        except Exception as e:
            logger.error(f"error generando el tables_master: {e}", exc_info=True)
        
        logger.info(f"Tabla debug generada de: {file_name}")
                        
        return output_file
    except Exception as e:
        logger.error(f"Error guardando CSV: {e}", exc_info=True)
        
def _append_table_to_master(corrected_df: pd.DataFrame, output_dir: str, section_title: str, header_text: List[str], master_filename: str = "tables_master.csv"):
    """
    Appendea una tabla a un único CSV maestro con secciones, manteniendo headers por tabla.
    Formato:
    """
    os.makedirs(output_dir, exist_ok=True)
    master_path = os.path.join(output_dir, master_filename)
    with open(master_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([f"# --- {section_title} ---"])
        writer.writerow(header_text if (header_text and len(header_text) > 0) else list(corrected_df.columns))
        for row in corrected_df.itertuples(index=False, name=None):
            writer.writerow(row)
        writer.writerow([])
