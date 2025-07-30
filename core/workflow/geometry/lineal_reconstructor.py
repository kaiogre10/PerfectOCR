# PerfectOCR/core/lineal_finder/line_reconstructor.py
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

    def _reconstruct_lines(self, polygons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Asigna un 'line_id' a cada polígono en la lista de entrada, enriqueciéndola."""

        prepared_sorted = sorted(polygons, key=lambda p: p.get("geometry", {}).get("centroid", [0, 0])[1])
        current_line_polys = []
        current_line_bbox = None
        line_counter = 0

        for poly in prepared_sorted:
            poly_centroid_y = poly.get("geometry", {}).get("centroid", [0, 0])[1]
            poly_bbox = poly.get("geometry", {}).get("bounding_box")

            if not poly_bbox:
                logger.debug(f"Polígono {poly.get('polygon_id')} descartado por falta de Bbox.")
                continue

            if not current_line_polys:
                current_line_polys = [poly]
                current_line_bbox = list(poly_bbox)
            else:
                # Añadimos una guarda para mayor seguridad
                if current_line_bbox:
                    line_ymin, line_ymax = current_line_bbox[1], current_line_bbox[3]
                    drops_intervall = (line_ymin <= poly_centroid_y <= line_ymax)
                    
                    # Calcular solapamiento vertical
                    y1_min, y1_max = current_line_bbox[1], current_line_bbox[3]
                    y2_min, y2_max = poly_bbox[1], poly_bbox[3]
                    overlap_abs = max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))
                    min_h = min(y1_max - y1_min, y2_max - y2_min)
                    overlap = overlap_abs / min_h if min_h > 1e-5 else 0.0

                    if drops_intervall and overlap > 0.3:
                        current_line_polys.append(poly)
                        # Actualizar el bbox de la línea actual
                        all_bboxes = [p.get("geometry", {}).get("bounding_box") for p in current_line_polys if p.get("geometry", {}).get("bounding_box")]
                        if all_bboxes:
                            xs = [b[0] for b in all_bboxes] + [b[2] for b in all_bboxes]
                            ys = [b[1] for b in all_bboxes] + [b[3] for b in all_bboxes]
                            current_line_bbox = [min(xs), min(ys), max(xs), max(ys)]
                    else:
                        # Asignar line_id a la línea completada
                        line_id = f"line_{line_counter:04d}"
                        for p in current_line_polys:
                            p['line_id'] = line_id
                        
                        line_counter += 1
                        current_line_polys = [poly]
                        current_line_bbox = list(poly_bbox)
                else:
                    # Si no hay bbox de línea, empezamos una nueva con el polígono actual
                    current_line_polys = [poly]
                    current_line_bbox = list(poly_bbox)

        # Asignar line_id a la última línea
        if current_line_polys:
            line_id = f"line_{line_counter:04d}"
            for p in current_line_polys:
                p['line_id'] = line_id

        logger.info(f"Asignación completada: {len(polygons)} polígonos asignados a {line_counter + 1} líneas lógicas.")
        
        lineal_polygons = polygons
        return lineal_polygons