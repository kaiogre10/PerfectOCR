# PerfectOCR/core/workflow/preprocessing/poly_gone.py
import cv2
import numpy as np
import logging
import os
from typing import Tuple, List, Dict, Any
from core.workspace.utils.output_handlers import ImageOutputHandler

logger = logging.getLogger(__name__)

class PolygonExtractor:
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.config = config
        self.image_saver = ImageOutputHandler()

    def _add_cropped_images_to_lines(self, deskewed_img: np.ndarray, reconstructed_lines: List[Dict], metadata: Dict, input_filename: str = "", output_config: Dict = None): # type: ignore
        """
        Añade las imágenes recortadas de cada línea al diccionario correspondiente y opcionalmente las guarda.
        También procesa polígonos individuales y los devuelve por separado.
        """
        if not reconstructed_lines:
            return {"lines": [], "metadata": metadata}, []

        img_h, img_w = deskewed_img.shape[:2]
        padding = self.config.get('cropping_padding', 5)
        
        # Verificar si el guardado de imágenes está habilitado
        should_save_line_images = False
        should_save_polygon_images = False
        if output_config:
            enabled_outputs = output_config.get('enabled_outputs', {})
            should_save_line_images = enabled_outputs.get('cropped_line_images', False)
            should_save_polygon_images = enabled_outputs.get('cropped_words', False)

        # Directorios de salida
        line_output_folder = None
        polygon_output_folder = None
        base_name = "imagen"
        
        if should_save_line_images:
            line_output_folder = os.path.join(self.project_root, "output", "cropped_lines")
            os.makedirs(line_output_folder, exist_ok=True)
            
        if should_save_polygon_images:
            polygon_output_folder = os.path.join(self.project_root, "output", "cropped_words")
            os.makedirs(polygon_output_folder, exist_ok=True)
            
        # Obtener nombre base del archivo
        if input_filename:
            base_name = os.path.splitext(os.path.basename(input_filename))[0]

        # Lista para almacenar polígonos individuales
        individual_polygons = []

        # Procesar líneas
        for line_idx, line in enumerate(reconstructed_lines):
            try:
                # Obtener el bounding box de la línea
                min_x, min_y, max_x, max_y = map(int, line["line_bbox"])
                
                # Aplicar padding y asegurar límites de la imagen
                x1 = max(0, min_x - padding)
                y1 = max(0, min_y - padding)
                x2 = min(img_w, max_x + padding)
                y2 = min(img_h, max_y + padding)
                
                # Verificar que las coordenadas sean válidas
                if x2 > x1 and y2 > y1:
                    # Recortar la imagen de la línea
                    cropped_img = deskewed_img[y1:y2, x1:x2]
                    
                    if cropped_img.size > 0:
                        line["cropped_img"] = cropped_img
                        
                        # Guardar la imagen recortada solo si está habilitado
                        if should_save_line_images and line_output_folder:
                            line_number = line_idx + 1  # Empezar desde 1
                            output_filename = f"{base_name}_linea_{line_number:03d}.png"
                            saved_path = self.image_saver.save(cropped_img, line_output_folder, output_filename)
                            
                            if saved_path:
                                line["saved_image_path"] = saved_path
                        
                    else:
                        logger.warning(f"Imagen vacía para línea {line['line_id']}")
                        line["cropped_img"] = None
                else:
                    logger.warning(f"Coordenadas inválidas para línea {line['line_id']}: bbox={line['line_bbox']}")
                    line["cropped_img"] = None

                # Procesar polígonos individuales de esta línea
                if "polygons" in line:
                    for poly_idx, poly in enumerate(line["polygons"]):
                        try:
                            # Obtener el bounding box del polígono
                            poly_min_x, poly_min_y, poly_max_x, poly_max_y = map(int, poly["bbox"])
                            
                            # Aplicar padding y asegurar límites de la imagen
                            poly_x1 = max(0, poly_min_x - padding)
                            poly_y1 = max(0, poly_min_y - padding)
                            poly_x2 = min(img_w, poly_max_x + padding)
                            poly_y2 = min(img_h, poly_max_y + padding)
                            
                            # Verificar que las coordenadas sean válidas
                            if poly_x2 > poly_x1 and poly_y2 > poly_y1:
                                # Recortar la imagen del polígono
                                cropped_poly = deskewed_img[poly_y1:poly_y2, poly_x1:poly_x2]
                                
                                if cropped_poly.size > 0:
                                    # Crear diccionario del polígono individual con su propia metadata
                                    individual_poly = {
                                        "polygon_id": poly["polygon_id"],
                                        "line_id": line["line_id"],
                                        "line_idx": line_idx,
                                        "poly_idx": poly_idx,
                                        "bbox": poly["bbox"],
                                        "centroid": poly["centroid"],
                                        "height": poly["height"],
                                        "width": poly["width"],
                                        "coords": poly["coords"],
                                        "cropped_img": cropped_poly,
                                        "metadata": {
                                            "line_id": line["line_id"],
                                            "polygon_id": poly["polygon_id"],
                                            "processing_info": metadata
                                        }
                                    }
                                    
                                    # Guardar la imagen del polígono solo si está habilitado
                                    if should_save_polygon_images and polygon_output_folder:
                                        polygon_filename = f"{base_name}_linea_{line_idx + 1:03d}_poly_{poly_idx + 1:03d}.png"
                                        saved_poly_path = self.image_saver.save(cropped_poly, polygon_output_folder, polygon_filename)
                                        
                                        if saved_poly_path:
                                            individual_poly["saved_image_path"] = saved_poly_path
                                    
                                    individual_polygons.append(individual_poly)
                                    
                                else:
                                    logger.warning(f"Imagen vacía para polígono {poly['polygon_id']} en línea {line['line_id']}")
                            else:
                                logger.warning(f"Coordenadas inválidas para polígono {poly['polygon_id']}: bbox={poly['bbox']}")
                                
                        except Exception as e:
                            logger.error(f"Error recortando polígono {poly.get('polygon_id', 'desconocido')}: {e}")
                    
            except Exception as e:
                logger.error(f"Error recortando línea {line.get('line_id', 'desconocida')}: {e}")
                line["cropped_img"] = None

        # Logs informativos
        if should_save_line_images:
            logger.info(f"Guardado de imágenes de líneas habilitado: {len(reconstructed_lines)} imágenes procesadas")
        else:
            logger.debug("Guardado de imágenes de líneas deshabilitado")
            
        if should_save_polygon_images:
            logger.info(f"Guardado de imágenes de polígonos habilitado: {len(individual_polygons)} imágenes procesadas")
        else:
            logger.debug("Guardado de imágenes de polígonos deshabilitado")

        result = {
            "lines": reconstructed_lines,
            "metadata": metadata
        }
        return result, individual_polygons
        