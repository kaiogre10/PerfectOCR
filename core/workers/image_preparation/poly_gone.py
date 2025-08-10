# PerfectOCR/core/preprocessing/poly_gone.py
import numpy as np
import logging
from typing import Dict, Any, List
from core.factory.abstract_worker import AbstractWorker
from core.domain.data_formatter import DataFormatter

logger = logging.getLogger(__name__)

class PolygonExtractor(AbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.config = config

    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        try:
            full_img = context.get("full_img")
            if full_img is None:
                logger.error("PolygonExtractor: 'full_img' no encontrado en el contexto.")
                return False

            dict_data = manager.get_dict_data()
            polygons = dict_data.get("image_data", {}).get("polygons", {})
            if not isinstance(polygons, dict) or not polygons:
                logger.warning("PolygonExtractor: No se encontraron polígonos para procesar.")
                return True

            padding = int(self.config.get("cropping_padding", 5))
            img_h, img_w = full_img.shape[:2]

            # Asignar line_id usando la lógica del lineal_reconstructor
            line_id_map = self.assign_line_id(polygons)

            abstract_update: Dict[str, Dict[str, Any]] = {
                "cropped_img": {},
                "line_id": line_id_map
            }
            extracted_count = 0
            for poly_id, poly_data in polygons.items():
                bbox = (poly_data.get("geometry") or {}).get("bounding_box")
                if not isinstance(bbox, list) or len(bbox) != 4: continue

                x1, y1, x2, y2 = map(int, bbox)
                px1, py1 = max(0, x1 - padding), max(0, y1 - padding)
                px2, py2 = min(img_w, x2 + padding), min(img_h, y2 + padding)
                if px2 <= px1 or py2 <= py1: continue

                cropped = full_img[py1:py2, px1:px2].copy()
                if cropped.size == 0: continue

                abstract_update["cropped_img"][poly_id] = cropped
                extracted_count += 1

            if extracted_count > 0 and not manager.update_data(abstract_update):
                logger.error("PolygonExtractor: Fallo al guardar datos en el manager.")
                return False

            context["full_img"] = None
            manager.dict_id["full_img"] = None
            
            logger.info(f"PolygonExtractor: {extracted_count} recortes creados. 'full_img' liberada.")
            return True

        except Exception as e:
            logger.error(f"Error en PolygonExtractor: {e}", exc_info=True)
            return False

    def assign_line_id(self, polygons: Dict[str, Any]) -> Dict[str, str]:
        """
        Asigna un 'line_id' a cada polígono.
        Devuelve un mapeo polygon_id -> line_id.
        """
        if not polygons:
            logger.warning("No hay polígonos para asignar line_id.")
            return {}

        # Ordenar por la coordenada Y del centroide para simular líneas
        prepared_sorted = sorted(
            polygons.values(),
            key=lambda p: p.get("geometry", {}).get("centroid", [0, 0])[1]
        )
        id_map: Dict[str, str] = {}
        line_counter = 1
        last_centroid_y = None
        line_id = f"line_{line_counter:04d}"

        for poly in prepared_sorted:
            centroid = poly.get("geometry", {}).get("centroid")
            if centroid is None:
                continue
            centroid_y = centroid[1]
            if last_centroid_y is not None and abs(centroid_y - last_centroid_y) > 20:
                # Si la diferencia en Y es significativa, nueva línea
                line_counter += 1
                line_id = f"line_{line_counter:04d}"
            poly_id = poly.get("polygon_id")
            if poly_id:
                id_map[poly_id] = line_id
            last_centroid_y = centroid_y

        return id_map