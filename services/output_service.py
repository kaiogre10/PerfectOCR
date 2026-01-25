# core/utils/output_service.py
import os
import json
import logging
import numpy as np
import pandas as pd # type: ignore
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def save_shapes(image_name: str, poly_id: str, image: np.ndarray[Any, Any], output_paths: List[str] | str, contours1: List[np.ndarray[Any, Any]], contours2: List[np.ndarray[Any, Any]]):
    """Guarda una imagen con los contornos marcados sobre ella"""
    try:
        import cv2
        if isinstance(output_paths, str):
            output_paths = [output_paths]

        for path in output_paths:
            output_dir = os.path.join(path, image_name)
            file_name = f"{poly_id}.png"
            # Dibuja todos los contornos sobre la imagen
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) #type: ignore
            
            if not contours2:
                logger.info("Solo contornos principales")
                cv2.drawContours(image, [np.array(cont, dtype=np.int32) for cont in contours1], -1, (255, 0, 0), thickness=cv2.FILLED) # blobs
                save_image(image, output_dir, file_name)

            else:
                logger.info("Todos los contornos")
                cv2.drawContours(image, [np.array(cont, dtype=np.int32) for cont in contours1], -1, (0, 0, 255), thickness=1) # ruido rojo
                cv2.drawContours(image, [np.array(cont, dtype=np.int32) for cont in contours2], -1, (0, 255, 0), thickness=1) # corrección verde
                save_image(image, output_dir, file_name)

    except Exception as e:
        logger.error(f"Error guardando contornos: {e}", exc_info=True)

def save_croped_image(image_name: str, img_id: str, image: np.ndarray[Any, Any], output_paths: List[str] | str, worker_name: str): 
    """Guarda una imagen de depuración si la salida está habilitada."""
    if isinstance(output_paths, str):
        output_paths = [output_paths]

    for path in output_paths:
        output_dir = os.path.join(path, image_name)
        file_name = f"{img_id}.png"
        save_image(image, output_dir, file_name)
        output_dir = os.path.join(path, worker_name, image_name)

    logger.debug(f"Imagenes debug de {worker_name} guardadas")

def save_image(image: np.ndarray[Any, np.dtype[np.uint8]], output_dir: str, file_name: str):
    """Guarda una única imagen en disco."""
    try:
        import cv2
        os.makedirs(output_dir, exist_ok=True)
        img_path = os.path.join(output_dir, file_name)
        cv2.imwrite(img_path, image)
        
        return img_path
    except Exception as e:
        logger.error(f"Error guardando '{file_name}' imagen: {e}")
        
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

        logger.warning(f"JSON de {worker_name} generado para '{file_name}'.")

    except Exception as e:
        logger.warning(f"Error guardando {worker_name}.JSON: {e}", exc_info=True)
    
def save_raw_json(output_paths: List[str] | str, worker_name: str, results: Dict[str, Any], file_name: str) -> bool:
    try:
        if isinstance(output_paths, str):
            output_paths = [output_paths]

        for path in output_paths:
            output_dir = os.path.join(path, worker_name)
            file_name = f"{file_name}_{worker_name}.json"
            if save_json(results, output_dir, file_name):
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
        
def save_debug_table(corrected_df: pd.DataFrame, file_name: str, output_paths: List[str] | str, worker_name: str, header_polygons: List[Any]):
    try:
        header_text: List[str] = []
        for poly_obj in header_polygons:
            poly_text = getattr(poly_obj, "ocr_text", None)
            if poly_text:
                header_text.append(poly_text)
        if not header_text:
            header_text = list(corrected_df.columns)

        for path in output_paths:
            output_dir = os.path.join(path, worker_name)
            file_name = f"{file_name}_{worker_name}.csv"
            save_table(corrected_df, output_dir, file_name, header_text)

    except Exception as e:
        logger.error(f"Error guardadndo tabla JSON de {worker_name},: {e}", exc_info=True)

def save_table_values(file_name: str, all_features: Dict[str, Dict[str, float]], output_paths: List[str] | str, worker_name: str, image_features: bool):
    try:
        df: pd.DataFrame = pd.DataFrame.from_dict(all_features, orient='index') #type: ignore
        df.index.name = 'line_id'
        
        # Resetear índice para que line_id sea una columna
        df = df.reset_index()

        for path in output_paths:
            output_dir = os.path.join(path, worker_name)
            table_file_name = f"{file_name}_{worker_name}.csv"
            save_table(df, output_dir, table_file_name, list(df.columns))

        if image_features:
            import matplotlib.pyplot as plt
            features_data = df.drop('line_id', axis=1)
            feature_names: List[str] = list(features_data.columns.tolist()) # type: ignore
            
            # Crear la figura
            plt.figure(figsize=(12, 8)) #type: ignore
            
            # Plotear cada línea del documento con valores originales
            for idx, row in features_data.iterrows():
                line_id: str = df.iloc[idx]['line_id'] #type: ignore
                plt.plot(feature_names, row.values, label=f'Línea {line_id}', alpha=0.7, linewidth=1) #type: ignore
            
            # Configurar la gráfica
            plt.xlabel('Features') #type: ignore
            plt.ylabel('Valores de Features') #type: ignore
            plt.title(f'Comportamiento de Features por Línea - {os.path.splitext(file_name)[0]}')#type: ignore
            plt.xticks(rotation=45, ha='right') #type: ignore
            plt.grid(True, alpha=0.3) #type: ignore
            
            # Calcular los límites del eje Y y poner los ticks de 1 en 1
            if not features_data.empty:
                ymin = features_data.min().min()# type: ignore
                ymax = features_data.max().max()# type: ignore
                ymin_tick = int(np.floor(ymin))# type: ignore
                ymax_tick = int(np.ceil(ymax))# type: ignore
                plt.yticks(np.arange(ymin_tick, ymax_tick + 1, 1)) #type: ignore
            
            # Limitar leyenda si hay muchas líneas
            if len(df) > 20:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8) #type: ignore
            else:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') #type: ignore
            
            plt.tight_layout()
            
            # Guardar la gráfica
            plot_filename = f"{os.path.splitext(file_name)[0]}_features_graph.png"
            plot_path = os.path.join(output_dir, plot_filename) #type: ignore
            plt.savefig(plot_path, dpi=300, bbox_inches='tight') #type: ignore
            plt.close()
            
            logger.info(f"Gráfica de features guardada en: {plot_path}")
    except Exception as e:
        logger.error(f"Error calculando Features output: {e}", exc_info=True)
        
def save_table(corrected_df: pd.DataFrame, output_dir: str, file_name: str, header_text: List[str]):
    """
    Guarda una tabla estructurada en formato CSV (compatible con Excel).
    Ruta del archivo guardado o None si hay error.
    """
    import csv
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
    import csv
    os.makedirs(output_dir, exist_ok=True)
    master_path = os.path.join(output_dir, master_filename)
    with open(master_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([f"# --- {section_title} ---"])
        writer.writerow(header_text if (header_text and len(header_text) > 0) else list(corrected_df.columns))
        for row in corrected_df.itertuples(index=False, name=None):
            writer.writerow(row)
        writer.writerow([])
