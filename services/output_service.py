# core/utils/output_service.py
import os
import json
import cv2
import logging
import csv
import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, List
from core.domain.data_formatter import DataFormatter

logger = logging.getLogger(__name__)

def save_json(final_results: List[Dict[str, Any]], output_dir: str, file_name: str, project_root: str) -> Optional[str]:
    """Guarda un JSON en disco."""
    try:
        project_root = project_root
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, file_name)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=4, ensure_ascii=False)
        return output_file
    except Exception as e:
        logger.error(f"Error guardando JSON: {e}", exc_info=True)
        return None
    
def save_tabjson(line_ids: List[str], manager: DataFormatter, output_dir: str, file_name: str, project_root: str) -> Optional[str]:
    """Guarda un TABJSON en disco."""
    project_root = project_root
    try:
        marked_count = 0
        tabular_lines_info: List[Dict[str, Any]] = []
        marked_ids: List[str] = []
        for line_id in line_ids:
            if line_id in manager.workflow.all_lines if manager.workflow else {}:
                line_obj = manager.workflow.all_lines[line_id]
                line_obj.tabular_line = True
                manager.workflow.all_lines[line_id] = line_obj
                marked_count += 1
                marked_ids.append(line_id)
                tabular_lines_info.append({
                    "line_id": line_id,
                    "text": getattr(manager.workflow.all_lines[line_id], "text", ""),
                    "polygon_ids": getattr(manager.workflow.all_lines[line_id], "polygon_ids", [])
                })

        if marked_ids:
            logger.debug(f"Marcadas {marked_count} líneas como tabulares: {marked_ids}")
            for log_info in tabular_lines_info:
                logger.debug(f"Líneas tabulares: {log_info['line_id']}: '{log_info['text']}' | polygons: {log_info['polygon_ids']}")

    except Exception as e:
        logger.error(f"Error guardando JSON: {e}", exc_info=True)

        output_file = os.path.join(output_dir, file_name)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(tabular_lines_info, f, indent=4, ensure_ascii=False)
        return output_file
    

def save_image(image: np.ndarray[Any, np.dtype[np.uint8]], output_dir: str, file_name_with_extension: str) -> Optional[str]:
    """Guarda una única imagen en disco."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        img_path = os.path.join(output_dir, file_name_with_extension)
        cv2.imwrite(img_path, image)
        return img_path
    except Exception as e:
        logger.error(f"Error guardando imagen: {e}")
        return None

def save_text(text: str, output_dir: str, file_name_with_extension: str) -> Optional[str]:
    """Guarda texto en disco."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, file_name_with_extension)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        return output_file
    except Exception as e:
        logger.error(f"Error guardando texto: {e}")
        return None
        
def save_table(corrected_df: pd.DataFrame, output_dir: str, file_name: str, header_text: List[str]) -> Optional[str]:
    """
    Guarda una tabla estructurada en formato CSV (compatible con Excel).
    Args:
        corrected_df: DataFrame con los datos corregidos.
        output_dir: Carpeta de salida.
        file_name: Nombre del archivo CSV.
        line_header: Lista de nombres de columnas.
    Returns:
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
                        
        return output_file
    except Exception as e:
        logger.error(f"Error guardando CSV: {e}", exc_info=True)
        return None
        
def _append_table_to_master(corrected_df: pd.DataFrame, output_dir: str, section_title: str, header_text: List[str], master_filename: str = "tables_master.csv") -> Optional[str]:
    """
    Appendea una tabla a un único CSV maestro con secciones, manteniendo headers por tabla.
    Formato:
      # --- <section_title> ---
      <header>
      <rows...>
      <blank line>
    """
    os.makedirs(output_dir, exist_ok=True)
    master_path = os.path.join(output_dir, master_filename)
    with open(master_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([f"# --- {section_title} ---"])
        writer.writerow(header_text if (header_text and len(header_text) > 0) else list(corrected_df.columns))
        for row in corrected_df.itertuples(index=False, name=None):
            writer.writerow(row)
        writer.writerow([])  # separador entre tablas
    return master_path