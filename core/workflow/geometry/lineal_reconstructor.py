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

    def _reconstruct_lines(self, polygons: List[List[List[float]]], metadata: dict):
        """
        Asigna un 'line_id' a cada polígono agrupándolos lógicamente en líneas,
        y retorna una lista de polígonos, cada uno con su metadata individual.
        """
        logger.info(f"Iniciando asignación de line_id a {len(polygons)} polígonos")

        def _get_geometry(polygon_coords):
            xs = [p[0] for p in polygon_coords]
            ys = [p[1] for p in polygon_coords]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            cx = sum(xs) / len(xs) if xs else 0.0
            cy = sum(ys) / len(ys) if ys else 0.0
            return {
                "bbox": [min_x, min_y, max_x, max_y],
                "centroid": [cx, cy],
                "height": max_y - min_y,
                "width": max_x - min_x
            }

        def _vertical_overlap(bbox1, bbox2):
            y1_min, y1_max = bbox1[1], bbox1[3]
            y2_min, y2_max = bbox2[1], bbox2[3]
            h1 = y1_max - y1_min
            h2 = y2_max - y2_min
            if h1 <= 1e-5 or h2 <= 1e-5:
                return 0.0
            overlap_abs = max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))
            min_h = min(h1, h2)
            if min_h <= 1e-5:
                return 0.0
            return overlap_abs / min_h

        # Preparar polígonos con su geometría
        prepared = []
        for idx, poly_coords in enumerate(polygons):
            if not poly_coords or len(poly_coords) < 3:
                logger.debug(f"Polígono {idx} descartado: insuficientes puntos ({len(poly_coords) if poly_coords else 0})")
                continue
            geom = _get_geometry(poly_coords)
            prepared.append({
                "polygon_id": f"poly_{idx:04d}",
                "coords": poly_coords,
                **geom
            })

        logger.info(f"Polígonos válidos preparados: {len(prepared)}/{len(polygons)}")

        prepared_sorted = sorted(prepared, key=lambda p: p["centroid"][1])

        lineal_polygons = []
        current_line_polys = []
        current_line_bbox = None
        line_counter = 0

        for poly in prepared_sorted:
            poly_centroid_y = poly["centroid"][1]
            poly_bbox = poly["bbox"]
            if not current_line_polys:
                current_line_polys = [poly]
                current_line_bbox = poly_bbox.copy()
            else:
                line_ymin, line_ymax = current_line_bbox[1], current_line_bbox[3]
                drops_intervall = (line_ymin <= poly_centroid_y <= line_ymax)
                overlap = _vertical_overlap(current_line_bbox, poly_bbox)
                if drops_intervall and overlap > 0.3:
                    current_line_polys.append(poly)
                    xs = [p for poly2 in current_line_polys for p in [poly2["bbox"][0], poly2["bbox"][2]]]
                    ys = [p for poly2 in current_line_polys for p in [poly2["bbox"][1], poly2["bbox"][3]]]
                    current_line_bbox = [min(xs), min(ys), max(xs), max(ys)]
                else:
                    line_id = f"line_{line_counter:04d}"
                    for p in current_line_polys:
                        lineal_polygons.append({
                            "polygon_id": p["polygon_id"],
                            "coords": p["coords"],
                            "bbox": p["bbox"],
                            "centroid": p["centroid"],
                            "height": p["height"],
                            "width": p["width"],
                            "line_id": line_id,
                            "metadata": {
                                "line_id": line_id,
                                "polygon_id": p["polygon_id"],
                                "processing_info": metadata
                            }
                        })
                    line_counter += 1
                    current_line_polys = [poly]
                    current_line_bbox = poly_bbox.copy()

        if current_line_polys:
            line_id = f"line_{line_counter:04d}"
            for p in current_line_polys:
                lineal_polygons.append({
                    "polygon_id": p["polygon_id"],
                    "coords": p["coords"],
                    "bbox": p["bbox"],
                    "centroid": p["centroid"],
                    "height": p["height"],
                    "width": p["width"],
                    "line_id": line_id,
                    "metadata": {
                        "line_id": line_id,
                        "polygon_id": p["polygon_id"],
                        "processing_info": metadata
                    }
                })

        logger.info(f"Asignación completada: {len(lineal_polygons)} polígonos asignados a {line_counter + 1} líneas lógicas.")

        # La salida es una lista de polígonos, cada uno con su metadata individual
        return lineal_polygons