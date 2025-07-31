# PerfectOCR/core/polygonal/line_reconstructor.py
import logging
import math
import numpy as np
from typing import Dict, Any, List
from shapely.geometry import Polygon

logger = logging.getLogger(__name__)

class LineReconstructor:
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.corrections = config
        self.lines_info: Dict[str, Any] = {}

    def _reconstruct_lines(self, enriched_doc: Dict[str, Any]) -> Dict[str, str]:
        """
        Asigna un 'line_id' a cada polígono.
        Devuelve un mapeo polygon_id -> line_id.
        """
        polygons = enriched_doc.get("polygons", [])
        if not polygons:
            logger.warning("No hay polígonos en el diccionario")
            return {}
        
        prepared_sorted = sorted(polygons, key=lambda p: p.get("geometry", {}).get("centroid", [0, 0])[1])
        id_map: Dict[str, str] = {}
        self.lines_info: Dict[str, Any] = {} 
        current_line_polys: List[Dict[str, Any]] = []
        current_line_bbox = None
        line_counter = 1

        def _finalize_line(polys, bbox, line_num):
            """Función helper para finalizar una línea y recolectar su información."""
            if not polys:
                return
            
            line_id = f"line_{line_num:04d}"
            polygon_ids = []
            centroids_x = []
            centroids_y = []

            for p in polys:
                pid = p.get("polygon_id")
                if pid:
                    id_map[pid] = line_id
                    polygon_ids.append(pid)
                    centroid = p.get("geometry", {}).get("centroid")
                    if centroid:
                        centroids_x.append(centroid[0])
                        centroids_y.append(centroid[1])
            
            line_centroid = [np.mean(centroids_x), np.mean(centroids_y)] if centroids_x else [0, 0]

            self.lines_info[line_id] = {
                "bounding_box": bbox,
                "centroid": line_centroid,
                "polygon_ids": polygon_ids
            }

        for poly in prepared_sorted:
            centroid_y = poly.get("geometry", {}).get("centroid", [0, 0])[1]
            bbox = poly.get("geometry", {}).get("bounding_box")
            if not bbox:
                continue
            if not current_line_polys:
                current_line_polys = [poly]
                current_line_bbox = list(bbox)
            else:
                # Verificar solapamiento vertical
                if current_line_bbox is not None:
                    y1_min, y1_max = current_line_bbox[1], current_line_bbox[3]
                    y2_min, y2_max = bbox[1], bbox[3]
                    drops = (y1_min <= centroid_y <= y1_max)
                    overlap_abs = max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))
                    min_h = min(y1_max - y1_min, y2_max - y2_min)
                    overlap = overlap_abs / min_h if min_h > 1e-5 else 0.0
                else:
                    drops = False
                    overlap = 0.0

                if drops and overlap > 0.3:
                    current_line_polys.append(poly)
                    # Actualizar bbox de línea usando el promedio vertical
                    all_bboxes = [p.get("geometry", {}).get("bounding_box") for p in current_line_polys if p.get("geometry", {}).get("bounding_box")]
                    if all_bboxes:
                        all_xs = [b[0] for b in all_bboxes] + [b[2] for b in all_bboxes]
                        all_y_mins = [b[1] for b in all_bboxes]
                        all_y_maxs = [b[3] for b in all_bboxes]
                        
                        avg_y_min = sum(all_y_mins) / len(all_y_mins)
                        avg_y_max = sum(all_y_maxs) / len(all_y_maxs)
                        
                        current_line_bbox = [min(all_xs), avg_y_min, max(all_xs), avg_y_max]
                else:
                    _finalize_line(current_line_polys, current_line_bbox, line_counter)
                    line_counter += 1
                    current_line_polys = [poly]
                    current_line_bbox = list(bbox)

        # Asignar última línea
        if current_line_polys:
            _finalize_line(current_line_polys, current_line_bbox, line_counter)

        return id_map
        
    def _get_lines_geometry(self) -> Dict[str, Any]:
        """
        Método separado para obtener la geometría de las líneas.
        Debe ser llamado después de _reconstruct_lines.
        """
        return self.lines_info.copy() if hasattr(self, 'lines_info') else {}