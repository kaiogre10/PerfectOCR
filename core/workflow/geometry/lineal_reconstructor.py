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

    @staticmethod
    def _reconstruct_lines(polygons: List[List[List[float]]], metadata: dict):
        """
        Agrupa polígonos en líneas si el centroide cae en el intervalo vertical de la línea
        y además hay suficiente solapamiento vertical (>30% del menor alto).
        Devuelve una lista de líneas (con geometría y polígonos) y la metadata.
        """
        logger.info(f"Iniciando reconstrucción de líneas con {len(polygons)} polígonos")

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

        # Ordenar por Y del centroide
        prepared_sorted = sorted(prepared, key=lambda p: p["centroid"][1])

        reconstructed_lines = []
        current_line = []
        current_line_bbox = None
        line_counter = 0
        for poly in prepared_sorted:
            poly_centroid_y = poly["centroid"][1]
            poly_bbox = poly["bbox"]
            if not current_line:
                # Iniciar nueva línea
                current_line = [poly]
                current_line_bbox = poly_bbox.copy()
            else:
                # Verificar si el centroide cae en el intervalo vertical de la línea
                line_ymin, line_ymax = current_line_bbox[1], current_line_bbox[3]
                drops_intervall = (line_ymin <= poly_centroid_y <= line_ymax)
                # Verificar solapamiento vertical
                overlap = _vertical_overlap(current_line_bbox, poly_bbox)
                if drops_intervall and overlap > 0.3:
                    current_line.append(poly)
                    # Actualizar bbox de la línea
                    xs = [p for poly2 in current_line for p in [poly2["bbox"][0], poly2["bbox"][2]]]
                    ys = [p for poly2 in current_line for p in [poly2["bbox"][1], poly2["bbox"][3]]]
                    current_line_bbox = [min(xs), min(ys), max(xs), max(ys)]
                else:
                    # Guardar línea anterior
                    xs = [p for poly2 in current_line for p in [poly2["bbox"][0], poly2["bbox"][2]]]
                    ys = [p for poly2 in current_line for p in [poly2["bbox"][1], poly2["bbox"][3]]]
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    cx = sum([poly2["centroid"][0] for poly2 in current_line]) / len(current_line)
                    cy = sum([poly2["centroid"][1] for poly2 in current_line]) / len(current_line)
                    reconstructed_lines.append({
                        "line_id": f"line_{line_counter:04d}",
                        "line_bbox": [min_x, min_y, max_x, max_y],
                        "line_centroid": [cx, cy],
                        "line_height": max_y - min_y,
                        "line_width": max_x - min_x,
                        "polygons": [
                            {
                                "polygon_id": poly2["polygon_id"],
                                "bbox": poly2["bbox"],
                                "centroid": poly2["centroid"],
                                "height": poly2["height"],
                                "width": poly2["width"],
                                "coords": poly2["coords"]
                            }
                            for poly2 in current_line
                        ]
                    })
                    line_counter += 1
                    # Iniciar nueva línea
                    current_line = [poly]
                    current_line_bbox = poly_bbox.copy()
        if current_line:
            xs = [p for poly2 in current_line for p in [poly2["bbox"][0], poly2["bbox"][2]]]
            ys = [p for poly2 in current_line for p in [poly2["bbox"][1], poly2["bbox"][3]]]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            cx = sum([poly2["centroid"][0] for poly2 in current_line]) / len(current_line)
            cy = sum([poly2["centroid"][1] for poly2 in current_line]) / len(current_line)
            reconstructed_lines.append({
                "line_id": f"line_{line_counter:04d}",
                "line_bbox": [min_x, min_y, max_x, max_y],
                "line_centroid": [cx, cy],
                "line_height": max_y - min_y,
                "line_width": max_x - min_x,
                "polygons": [
                    {
                        "polygon_id": poly2["polygon_id"],
                        "bbox": poly2["bbox"],
                        "centroid": poly2["centroid"],
                        "height": poly2["height"],
                        "width": poly2["width"],
                        "coords": poly2["coords"]
                    }
                    for poly2 in current_line
                ]
            })

        logger.info(f"Reconstrucción completada: {len(reconstructed_lines)} líneas generadas")

        return reconstructed_lines, metadata
