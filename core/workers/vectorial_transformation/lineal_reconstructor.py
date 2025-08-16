# PerfectOCR/core/workers/vectorial_transformation/linal_reconstructor.py
import logging
import math
import time
import numpy as np
from typing import Dict, Any, List, Optional, Union
from shapely.geometry import Polygon
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.domain.data_formatter import DataFormatter
from shapely.ops import unary_union

logger = logging.getLogger(__name__)

class LinealReconstructor(VectorizationAbstractWorker):
    
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get('lineal', {})
        self.enabled_outputs = self.config.get("enabled_outputs", {})
        self.output = self.enabled_outputs.get("reconstructed_lines", False)
        
    def vectorize(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        
        try:
            start_time = time.time
            polygons = context.get("polygons", {})
            if not polygons:
                logger.warning("No hay polígonos para reconstruir líneas.")
                return False
                
            lines = self._reconstruct_lines(polygons)
                
        
    def _reconstruct_lines(self, polygons: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reconstruye líneas agrupando polígonos y devuelve un dict con la info completa de cada línea,
        incluyendo los textos OCR concatenados.
        """
        prepared_sorted = sorted(
            polygons.values(),
            key=lambda p: p.get("geometry", {}).get("centroid", [0, 0])[1]
        )
        lines_info: Dict[str, Any] = {}
        current_line_polys: List[Dict[str, Any]] = []
        current_line_bbox = None
        line_counter = 1

        for poly in prepared_sorted:
            bbox = poly.get("geometry", {}).get("bounding_box")
            if not bbox:
                continue

            if not current_line_polys or current_line_bbox is None:
                current_line_polys = [poly]
                current_line_bbox = list(bbox)
            else:
                y1_min, y1_max = current_line_bbox[1], current_line_bbox[3]
                y2_min, y2_max = bbox[1], bbox[3]
                overlap_abs = max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))
                min_h = min(y1_max - y1_min, y2_max - y2_min)
                overlap = overlap_abs / min_h if min_h > 1e-5 else 0.0

                if overlap > 0.3:
                    current_line_polys.append(poly)
                    all_bboxes = [p.get("geometry", {}).get("bounding_box") for p in current_line_polys if p.get("geometry", {}).get("bounding_box")]
                    if all_bboxes:
                        all_xs = [b[0] for b in all_bboxes] + [b[2] for b in all_bboxes]
                        all_y_mins = [b[1] for b in all_bboxes]
                        all_y_maxs = [b[3] for b in all_bboxes]
                        avg_y_min = sum(all_y_mins) / len(all_y_mins)
                        avg_y_max = sum(all_y_maxs) / len(all_y_maxs)
                        current_line_bbox = [min(all_xs), avg_y_min, max(all_xs), avg_y_max]
                else:
                    # Finaliza la línea actual y guarda la info
                    line_id = f"line_{line_counter:04d}"
                    polygon_ids = [p.get("polygon_id") for p in current_line_polys if p.get("polygon_id")]
                    centroids_x = [p.get("geometry", {}).get("centroid")[0] for p in current_line_polys if p.get("geometry", {}).get("centroid")]
                    centroids_y = [p.get("geometry", {}).get("centroid")[1] for p in current_line_polys if p.get("geometry", {}).get("centroid")]
                    texts = [p.get("ocr_text", "") for p in current_line_polys]

                    lines_info[line_id] = {
                        "bounding_box": current_line_bbox,
                        "centroid": [np.mean(centroids_x), np.mean(centroids_y)] if centroids_x else [0, 0],
                        "polygon_ids": polygon_ids,
                        "text": " ".join(texts).strip()
                    }

                    line_counter += 1
                    current_line_polys = [poly]
                    current_line_bbox = list(bbox)

        # Finaliza la última línea
        if current_line_polys:
            line_id = f"line_{line_counter:04d}"
            polygon_ids = [p.get("polygon_id") for p in current_line_polys if p.get("polygon_id")]
            centroids_x = [p.get("geometry", {}).get("centroid")[0] for p in current_line_polys if p.get("geometry", {}).get("centroid")]
            centroids_y = [p.get("geometry", {}).get("centroid")[1] for p in current_line_polys if p.get("geometry", {}).get("centroid")]
            texts = [p.get("ocr_text", "") for p in current_line_polys]

            lines_info[line_id] = {
                "bounding_box": current_line_bbox,
                "centroid": [np.mean(centroids_x), np.mean(centroids_y)] if centroids_x else [0, 0],
                "polygon_ids": polygon_ids,
                "text": " ".join(texts).strip()
            }

        self.lines_info = lines_info
        return lines_info     
        
    def _get_lines_geometry(self) -> Dict[str, Any]:
        """
        Devuelve la geometría de las líneas construida.
        """
    return self.lines_info.copy()
