# PerfectOCR/core/preprocessing/poly_gone.py
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
        self.polygons_info: Dict[str, Any] = {}
        
    def _extract_individual_polygons(self, deskewed_img: np.ndarray, enriched_doc: Dict[str, Any]) -> Dict[str, Any]:
        """Extrae y recorta los polígonos individuales."""
        polygons = enriched_doc.get("polygons", {})
        if not polygons:
            logger.warning("No se encontraron polígonos en el documento enriquecido")
            return {}
        
        padding = self.config.get('cropping_padding', 5)
        if not polygons:
            return {}
        metadata = enriched_doc.get("metadata", {})
        img_dims = metadata.get("img_dims", {})
        img_h = int(img_dims.get("height", 0) or 0)
        img_w = int(img_dims.get("width", 0) or 0)

        # Inicializar el diccionario de información de polígonos
        self.polygons_info = {}
        
        for poly in polygons.values():
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
                        poly["padding_coords"] = [poly_x1, poly_y1, poly_x2, poly_y2]

                        polygon_id = poly.get("polygon_id")
                        if polygon_id:
                            self.polygons_info[polygon_id] = {
                                "polygon_coords": bbox.copy(),  # Copia de las coordenadas
                                "cropped_img": cropped_poly.copy(),  # Copia de la imagen
                                "polygon_id": polygon_id  # Copia del ID
                            }
                    else:
                        logger.warning(f"Imagen vacía para polígono {poly.get('polygon_id')}")
                        poly["cropped_img"] = None
                else:
                    logger.warning(f"Coordenadas inválidas para polígono {poly.get('polygon_id')}: bbox={bbox}")
                    poly["cropped_img"] = None
            except Exception as e:
                logger.error(f"Error recortando polígono {poly.get('polygon_id', 'desconocido')}: {e}")
                poly["cropped_img"] = None

        cropped_count = sum(1 for p in polygons.values() if p.get("cropped_img") is not None)
        logger.info(f"Imágenes recortadas exitosamente: {cropped_count}/{len(polygons)}")

        extracted_polygons = polygons
    
        return extracted_polygons
    
    def _get_polygons_copy(self) -> Dict[str, Any]:
        """
        Método separado para obtener la información específica de los polígonos.
        Debe ser llamado después de _extract_individual_polygons.
        """
        polygons_to_binarize =  self.polygons_info.copy() if hasattr(self, 'polygons_info') else {}
        return polygons_to_binarize