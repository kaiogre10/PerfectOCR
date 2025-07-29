# PerfectOCR/core/workflow/preprocessing/poly_gone.py
import cv2
import numpy as np
import logging
import os
from typing import Tuple, List, Dict, Any

logger = logging.getLogger(__name__)

class PolygonExtractor:
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.config = config
        cutter_params = self.config.get('cutting', {})
        self.padding = cutter_params.get('cropping_padding', {})
        
    def _extract_individual_polygons(self, deskewed_img: np.ndarray, lineal_polygons: List[Dict], input_filename: str = "", output_config: Dict = None) -> List[Dict[str, Any]]:
        """
        Extrae y recorta solo los polígonos individuales, asignando el line_id correspondiente.
        Devuelve una lista de polígonos, cada uno con su imagen recortada y metadata individual.
        """
        padding = self.config.get('cropping_padding', 5)
        if not lineal_polygons:
            return []

        img_h, img_w = deskewed_img.shape[:2]
        img_dims = img_h, img_w
        

        # Configuración de guardado de imágenes
        should_save_polygon_images = False
        polygon_output_folder = None
        base_name = "imagen"

        if output_config:
            enabled_outputs = output_config.get('enabled_outputs', {})
            should_save_polygon_images = enabled_outputs.get('cropped_words', False)

        if should_save_polygon_images:
            polygon_output_folder = os.path.join(self.project_root, "output", "cropped_words")
            os.makedirs(polygon_output_folder, exist_ok=True)

        if input_filename:
            base_name = os.path.splitext(os.path.basename(input_filename))[0]

        individual_polygons = []

        for poly in lineal_polygons:
            try:
                poly_min_x, poly_min_y, poly_max_x, poly_max_y = map(int, poly["bbox"])
                poly_x1 = max(0, poly_min_x - padding)
                poly_y1 = max(0, poly_min_y - padding)
                poly_x2 = min(img_w, poly_max_x + padding)
                poly_y2 = min(img_h, poly_max_y + padding)

                if poly_x2 > poly_x1 and poly_y2 > poly_y1:
                    cropped_poly = deskewed_img[poly_y1:poly_y2, poly_x1:poly_x2]
                    if cropped_poly.size > 0:
                        poly_dict = {
                            "line_id": poly["line_id"],
                            "polygon_id": poly["polygon_id"],
                            "coords": poly["coords"],
                            "bbox": poly["bbox"],
                            "centroid": poly["centroid"],
                            "height": poly["height"],
                            "width": poly["width"],
                            "cropped_img": cropped_poly,
                            "metadata": poly.get("metadata", {})
                        }
                        if should_save_polygon_images and polygon_output_folder:
                            polygon_filename = f"{base_name}_linea_{poly['line_id']}_poly_{poly['polygon_id']}.png"
                            saved_poly_path = self.image_saver.save(cropped_poly, polygon_output_folder, polygon_filename)
                            if saved_poly_path:
                                poly_dict["saved_image_path"] = saved_poly_path
                        individual_polygons.append(poly_dict)
                    else:
                        logger.warning(f"Imagen vacía para polígono {poly['polygon_id']} en línea {poly['line_id']}")
                else:
                    logger.warning(f"Coordenadas inválidas para polígono {poly['polygon_id']}: bbox={poly['bbox']}")
            except Exception as e:
                logger.error(f"Error recortando polígono {poly.get('polygon_id', 'desconocido')}: {e}")

        if should_save_polygon_images:
            logger.info(f"Total de: {len(individual_polygons)} imágenes de polígonos procesadas")
        else:
            logger.debug("Guardado de imágenes de polígonos deshabilitado")

        if isinstance(individual_polygons, list):
            lines_with_images = sum(1 for line in individual_polygons if line.get("cropped_img") is not None)
            logger.info(f"Imágenes recortadas exitosamente: {lines_with_images}/{len(individual_polygons)}")

        return individual_polygons
