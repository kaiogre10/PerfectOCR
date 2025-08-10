# PerfectOCR/core/preprocessing/poly_gone.py
import numpy as np
import logging
from typing import Dict, Any, List
from core.factory.abstract_worker import AbstractWorker
from core.domain.data_manager import DataFormatter

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

            sorted_polygons = sorted(
                polygons.items(), 
                key=lambda item: (item[1].get("geometry", {}).get("centroid", [0, 99999])[1])
            )

            line_id_map: Dict[str, str] = {}
            current_line_bbox = None
            line_counter = 1

            for poly_id, poly_data in sorted_polygons:
                bbox = (poly_data.get("geometry") or {}).get("bounding_box")
                if not bbox: continue

                if current_line_bbox is None:
                    line_id = f"line_{line_counter:04d}"
                    current_line_bbox = list(bbox)
                else:
                    y1_min, y1_max = current_line_bbox[1], current_line_bbox[3]
                    y2_min, y2_max = bbox[1], bbox[3]
                    overlap = max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))
                    min_h = min(y1_max - y1_min, y2_max - y2_min)
                    overlap_ratio = overlap / min_h if min_h > 1e-5 else 0.0

                    if overlap_ratio > 0.3:
                        line_id = f"line_{line_counter:04d}"
                        all_y = [current_line_bbox[1], current_line_bbox[3], bbox[1], bbox[3]]
                        current_line_bbox[1] = min(all_y)
                        current_line_bbox[3] = max(all_y)
                    else:
                        line_counter += 1
                        line_id = f"line_{line_counter:04d}"
                        current_line_bbox = list(bbox)
                
                line_id_map[poly_id] = line_id

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

    def _reconstruct_lines(self, doc_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Asigna un 'line_id' a cada polígono y construye la geometría de las líneas.
        Devuelve un mapeo polygon_id -> line_id.
        """
        polygons = doc_data.get("polygons", {})
        if not polygons:
            logger.warning("No hay polígonos para reconstruir líneas.")
            return {}
        
        prepared_sorted = sorted(polygons.values(), key=lambda p: p.get("geometry", {}).get("centroid", [0, 0])[1])
        id_map: Dict[str, str] = {}
        self.lines_info: Dict[str, Any] = {} 
        current_line_polys: List[Dict[str, Any]] = []
        current_line_bbox = None
        line_counter = 1

        def _finalize_line(polys, bbox, line_num):
            if not polys: return
            
            line_id = f"line_{line_num:04d}"
            polygon_ids = [p.get("polygon_id") for p in polys if p.get("polygon_id")]
            centroids_x = [p.get("geometry", {}).get("centroid")[0] for p in polys if p.get("geometry", {}).get("centroid")]
            centroids_y = [p.get("geometry", {}).get("centroid")[1] for p in polys if p.get("geometry", {}).get("centroid")]

            for pid in polygon_ids:
                id_map[pid] = line_id
            
            line_centroid = [np.mean(centroids_x), np.mean(centroids_y)] if centroids_x else [0, 0]
            
            self.lines_info[line_id] = {
                "bounding_box": bbox,
                "centroid": line_centroid,
                "polygon_ids": polygon_ids
            }

        for poly in prepared_sorted:
            bbox = poly.get("geometry", {}).get("bounding_box")
            if not bbox: continue

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
                    _finalize_line(current_line_polys, current_line_bbox, line_counter)
                    line_counter += 1
                    current_line_polys = [poly]
                    current_line_bbox = list(bbox)

        if current_line_polys:
            _finalize_line(current_line_polys, current_line_bbox, line_counter)

        return id_map