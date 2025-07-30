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
        
    def _extract_individual_polygons(self, deskewed_img: np.ndarray, lineal_polygons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extrae y recorta los polígonos individuales, añadiendo la 'cropped_img' a cada
        diccionario de polígono en la lista de entrada.
        """
        padding = self.config.get('cropping_padding', 5)
        if not lineal_polygons:
            return []

        img_h, img_w = deskewed_img.shape[:2]
        
        for poly in lineal_polygons:
            try:
                # La geometría ahora está anidada
                bbox = poly.get("geometry", {}).get("bounding_box")
                if not bbox:
                    logger.warning(f"Polígono {poly.get('polygon_id')} no tiene Bbox. Omitiendo.")
                    continue

                poly_min_x, poly_min_y, poly_max_x, poly_max_y = map(int, bbox)
                poly_x1 = max(0, poly_min_x - padding)
                poly_y1 = max(0, poly_min_y - padding)
                poly_x2 = min(img_w, poly_max_x + padding)
                poly_y2 = min(img_h, poly_max_y + padding)

                if poly_x2 > poly_x1 and poly_y2 > poly_y1:
                    cropped_poly = deskewed_img[poly_y1:poly_y2, poly_x1:poly_x2]
                    if cropped_poly.size > 0:
                        poly["cropped_img"] = cropped_poly
                        # Añadir dimensiones reales del polígono a geometry
                        poly_height, poly_width = cropped_poly.shape[:2]
                        poly.setdefault("geometry", {})["height"] = poly_height
                        poly.setdefault("geometry", {})["width"] = poly_width
                    else:
                        logger.warning(f"Imagen vacía para polígono {poly.get('polygon_id')}")
                        poly["cropped_img"] = None
                else:
                    logger.warning(f"Coordenadas inválidas para polígono {poly.get('polygon_id')}: bbox={bbox}")
                    poly["cropped_img"] = None
            except Exception as e:
                logger.error(f"Error recortando polígono {poly.get('polygon_id', 'desconocido')}: {e}")
                poly["cropped_img"] = None

        cropped_count = sum(1 for p in lineal_polygons if p.get("cropped_img") is not None)
        logger.info(f"Imágenes recortadas exitosamente: {cropped_count}/{len(lineal_polygons)}")

        extracted_polygons = lineal_polygons
    
        return extracted_polygons
