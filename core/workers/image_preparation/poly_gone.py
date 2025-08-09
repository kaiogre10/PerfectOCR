# PerfectOCR/core/preprocessing/poly_gone.py
import numpy as np
import logging
from typing import Dict, Any
from core.workers.factory.abstract_worker import AbstractWorker

logger = logging.getLogger(__name__)

class PolygonExtractor(AbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.config = config

    def process(self, image: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        dict_data = context["dict_data"]
        img_h = dict_data["metadata"]["img_dims"]["height"]
        img_w = dict_data["metadata"]["img_dims"]["width"]

        # Asegurar estructura
        if "image_data" not in dict_data:
            dict_data["image_data"] = {}
        if "polygons" not in dict_data["image_data"]:
            dict_data["image_data"]["polygons"] = {}

        polygons = dict_data["image_data"]["polygons"]
        if not polygons:
            logger.warning("[PolygonExtractor] No hay polígonos para recortar")
            return image

        padding = int(self.config.get("cropping_padding", 5))
        extracted = 0

        for poly_id, poly in polygons.items():
            geometry = poly.get("geometry")
            bbox = None
            if isinstance(geometry, dict):
                bbox = geometry.get("bounding_box")
            if not isinstance(bbox, (list)) or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = map(int, bbox)

            # Padding duro con clamp a bordes
            px1 = max(0, x1 - padding)
            py1 = max(0, y1 - padding)
            px2 = min(img_w, x2 + padding)
            py2 = min(img_h, y2 + padding)
            if px2 <= px1 or py2 <= py1:
                continue

            cropped = image[py1:py2, px1:px2]
            if cropped.size == 0:
                continue

            # Medidas reales del recorte
            width = float(px2 - px1)
            height = float(py2 - py1)
            padd_centroid = [px1 + width / 2.0, py1 + height / 2.0]
            padding_coords = [px1, py1, px2, py2]
            padding_bbox = [px1, py1, px2, py2]

            # Escribir salida completa
            poly["cropped_img"] = cropped
            if "cropedd_geometry" not in poly:
                poly["cropedd_geometry"] = {}

            poly["cropedd_geometry"].update({
                "padding_coords": padding_coords,   # [x1,y1,x2,y2] absolutos en la página
                "padding_bbox": padding_bbox,       # alias explícito del bbox del recorte
                "padd_centroid": padd_centroid,     # centroide del recorte
            })

            # Opcional: reflejar width/height del recorte
            # Si deseas que queden en geometry del polígono original:
            geom = poly.setdefault("cropedd_geometry", {})
            geom["width"] = width
            geom["height"] = height

            extracted += 1

        logger.info(f"[PolygonExtractor] Generadas padding-geometry y recortes para {extracted} polígonos")
        return image