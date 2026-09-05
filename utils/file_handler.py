import cv2
import numpy as np
import pandas as pd # type: ignore
import commentjson # type: ignore
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq
import pickle
import os
import csv
from typing import Any, Dict, List
from services.log_service import basic_exc_logger, log_simple

def load_images(image_path: str):
    if not image_path or not os.path.isfile(image_path):
        raise FileNotFoundError(f"No se proporcionó una ruta de entrada válida: '{image_path}'")
    return cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

def save_image(data: np.ndarray[Any, np.dtype[np.uint8]], output_dir: str, file_name: str):
    """Guarda una única imagen en disco."""
    try:    
        os.makedirs(output_dir, exist_ok=True)
        img_path = os.path.join(output_dir, file_name)
        cv2.imwrite(img_path, data)
    except Exception as e:
        basic_exc_logger(f"Error guardando '{file_name}' imagen: {e}")
    
def load_pickle(pkl_path: str, mode: str):
    """Carga pickle"""
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Pickle no encontrado en {pkl_path}")
    with open(pkl_path, mode) as f:
        model_pkl = pickle.load(f)
        if not model_pkl:
            raise pickle.UnpicklingError("ERROR EN LA CARGA DEL PICKLE")
    model_pkl["allow_edit"] = False
    return model_pkl

def save_pickle(model_pkl: Any, pkl_path: str, mode: str):
    model_pkl["allow_edit"] = False
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Ruta inválida para guardar pickle: {pkl_path}")
    with open(pkl_path, mode) as f:
        pickle.dump(model_pkl, f, protocol=5)

def load_yaml(file_path: str, mode: str):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"ARCHIVO DE CONFIGURACIÓN NO ENCONTRADO: {file_path}")
    
    yaml = YAML(typ='safe', pure=True)
    yaml.default_flow_style = False
    yaml.allow_unicode = True

    with open(file_path, mode, encoding='utf-8') as f:
        yaml_raw = yaml.load(f)
        if not yaml_raw:
            raise ValueError(f"YAML INEXISTENTE")
    return yaml_raw

def save_yaml(results: Dict[str, Dict[str, Any]], output_dir: str, file_name: str) -> bool:
    """Guarda un YAML en disco."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, file_name)

        # Convertir listas a formato inline
        for item in results.values():
            for key, value in item.items():
                if isinstance(value, list):
                    seq = CommentedSeq(value)
                    seq.fa.set_flow_style()  # <- [1, 1]
                    item[key] = seq

        yaml = YAML()
        yaml.default_flow_style = False
        yaml.allow_unicode = True

        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(results, f)

        return True

    except Exception as e:
        basic_exc_logger(f"Error guardando YAML: {e}", exc_info=True)
    return False

def load_jsoncomment(file_path: str, mode: str):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"ARCHIVO DE CONFIGURACIÓN NO ENCONTRADO: {file_path}")
    with open(file_path, mode, encoding='utf-8') as f:
        commentjson_raw = commentjson.load(f) # type: ignore
        if not commentjson_raw:
            raise ValueError(f"COMMENT JSON INEXISTENTE")
    return commentjson_raw

def save_table(corrected_df: pd.DataFrame, output_dir: str, file_name: str, stack: bool):
    """Guarda una tabla estructurada en formato CSV (compatible con Excel)."""
    try:
        header_text = list(corrected_df.columns)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, file_name)
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header_text)
            # Escribimos las filas del DataFrame, no solo los nombres de columnas
            for row in corrected_df.itertuples(index=False, name=None):
                writer.writerow(row)                
        if stack:
            try:
                append_table_to_master(
                    corrected_df=corrected_df,
                    output_dir=output_dir,
                    section_title=os.path.splitext(os.path.basename(file_name))[0],
                    header_text=header_text,
                    master_filename="tables_master.csv"
                )
            except Exception as e:
                basic_exc_logger(f"error generando el tables_master: {e}", exc_info=True)
        
        log_simple(f"Tabla generada de: '{file_name}'")
                        
        return output_file
    except Exception as e:
        basic_exc_logger(f"Error guardando CSV: {e}", exc_info=True)

def append_table_to_master(corrected_df: pd.DataFrame, output_dir: str, section_title: str, header_text: List[str], master_filename: str = "tables_master.csv"):
    os.makedirs(output_dir, exist_ok=True)
    master_path = os.path.join(output_dir, master_filename)
    write_header = not os.path.exists(master_path) or os.path.getsize(master_path) == 0

    with open(master_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([f"# --- {section_title} ---"])
        if write_header:
            writer.writerow(header_text if (header_text and len(header_text) > 0) else list(corrected_df.columns))
        for row in corrected_df.itertuples(index=False, name=None):
            writer.writerow(row)
