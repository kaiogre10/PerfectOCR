# PerfectOCR/core/preprocessing/poly_gone.py
import numpy as np
import logging
from typing import Dict, Any
from core.workers.factory.abstract_worker import AbstractWorker
from core.domain.workflow_manager import DataFormatter

logger = logging.getLogger(__name__)

class PolygonExtractor(AbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.config = config

    def process(self, image: np.ndarray, context: Dict[str, Any]) -> np.ndarray:  # type: ignore
        """
        Recolecta datos por polígono y envía un abstract ID-céntrico al manager:
        {
          "poly_0001": {
            "cropped_img": np.ndarray,
            "padding_coords": [x1, y1, x2, y2],
            "padding_bbox": [x1, y1, x2, y2],
            "padd_centroid": [cx, cy],
          },
          ...
        }
        """
        dict_data: Dict[str, Any] = context.get("dict_id", {})
        img_h: int = dict_data.get("metadata", {}).get("img_dims", {}).get("height", 0)
        img_w: int = dict_data.get("metadata", {}).get("img_dims", {}).get("width", 0)

        polygons: Dict[str, Any] = dict_data.get("image_data", {}).get("polygons", {}) or {}

        abstract: Dict[str, Dict[str, Any]] = {}
        padding: int = int(self.config.get("cropping_padding", 5))
        extracted: int = 0

        for poly_id, poly in polygons.items():
            geometry = poly.get("geometry") if isinstance(poly, dict) else None
            bbox = geometry.get("bounding_box") if isinstance(geometry, dict) else None
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = map(int, bbox)

            px1 = max(0, x1 - padding)
            py1 = max(0, y1 - padding)
            px2 = min(img_w, x2 + padding)
            py2 = min(img_h, y2 + padding)
            if px2 <= px1 or py2 <= py1:
                continue

            cropped = image[py1:py2, px1:px2]
            if getattr(cropped, "size", 0) == 0:
                continue

            padd_coords = [px1, py1, px2, py2]
            padd_bbox = [px1, py1, px2, py2]
            width = float(px2 - px1)
            height = float(py2 - py1)
            padd_cent = [px1 + width / 2.0, py1 + height / 2.0]

            abstract_poly = abstract.setdefault(poly_id, {})
            abstract_poly["cropped_img"] = cropped          # np.ndarray por referencia
            abstract_poly["padding_coords"] = padd_coords   # ints
            abstract_poly["padding_bbox"] = padd_bbox       # floats válidos también
            abstract_poly["padd_centroid"] = padd_cent      # floats

            extracted += 1

        formatter = DataFormatter()
        # Enlazar al dict vivo para escribir in-place
        formatter.dict_id = dict_data
        success = formatter.update_data(abstract)
        if not success:
            logger.error("[PolygonExtractor] Error actualizando manager con abstract ID-céntrico")

        logger.info(f"[PolygonExtractor] Recolectados datos para {extracted} polígonos.")
        # El worker devuelve la imagen de entrada (pipeline continúa)
        return image