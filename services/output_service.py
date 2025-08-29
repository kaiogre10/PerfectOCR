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
from core.domain.data_models import Metadata, AllLines

logger = logging.getLogger(__name__)

def save_json(data: Dict[str, Any], output_dir: str, file_name_with_extension: str) -> Optional[str]:
    """Guarda un JSON en disco."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, file_name_with_extension)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return output_file
    except Exception as e:
        logger.error(f"Error guardando JSON: {e}")
        return None

def save_image(image: np.ndarray[Any, Any], output_dir: str, file_name_with_extension: str) -> Optional[str]:
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

def save_ocr_json(manager: DataFormatter, output_folder: str, image_name: str) -> None:
    """
    Guarda los resultados OCR en formato JSON, usando la lógica centralizada.
    """
    try:
        if not manager.workflow:
            logger.warning("No se puede guardar resultados OCR porque el workflow_dict no está inicializado.")
            return

        # Preparar la estructura de datos para el JSON
        metadata: Dict[str, Metadata] = manager.workflow.metadata if manager.workflow else {}
        output_data: Dict[ str, Any] = {
            "doc_name": metadata.get("image_name"),
            "formato": metadata.get("format"),
            "dpi": metadata.get("dpi"),
            "img_dims": metadata.get("img_dims"),
            "fecha_creacion": str(metadata.get("date_creation")),
            "polygons": []
        }

        polygons_data = manager.workflow.polygons if manager.workflow else {}
        for pid, p_data in polygons_data.items():
            output_data["polygons"].append({
                "polygon_id": pid,
                "text": p_data.ocr_text,
                "confidence": p_data.ocr_confidence
            })

        # Usar la función save_json existente
        json_filename = f"{image_name}_ocr_results.json"
        save_json(output_data, output_folder, json_filename)
        logger.debug(f"Resultados OCR en JSON guardados en: {os.path.join(output_folder, json_filename)}")

    except Exception as e:
        logger.error(f"Error guardando resultados OCR en JSON: {e}")
        

def save_table(corrected_df: pd.DataFrame, output_dir: str, file_name: str, header_line: List[str]) -> Optional[str]:
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
            writer.writerow(header_line)
            # Escribimos las filas del DataFrame, no solo los nombres de columnas
            for row in corrected_df.itertuples(index=False, name=None):
                writer.writerow(row)
        return output_file
    except Exception as e:
        logger.error(f"Error guardando CSV: {e}")
        return None