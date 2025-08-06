# PerfectOCR/core/polygonal/line_reconstructor.py
import logging
import math
import numpy as np
from typing import Dict, Any, List
from shapely.geometry import Polygon
from core.workers.factory.abstract_worker import AbstractWorker
from core.domain.workflow_job import ProcessingStage

logger = logging.getLogger(__name__)

class LineReconstructor(AbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.corrections = config
        self.lines_info: Dict[str, Any] = {}

    def process(self, image: np.ndarray[Any, Any], context: Dict[str, Any]) -> np.ndarray[Any, Any]:
        """
        Implementa el método abstracto de AbstractWorker.
        """
        workflow_job = context.get('workflow_job')
        metadata = context.get('metadata', {})
        
        # Crear doc_data para compatibilidad
        doc_data = {
            "metadata": metadata,
            "polygons": {}  # Los polígonos vendrían del worker anterior
        }
        
        # Reconstruir líneas
        id_map = self._reconstruct_lines(doc_data)
        
        # Actualizar el WorkflowJob si está disponible
        if workflow_job and workflow_job.full_img is not None:
            workflow_job.update_stage(ProcessingStage.LINES_RECONSTRUCTED)
        
        return image  # Retorna la misma imagen (no la modifica)

    def _reconstruct_lines(self, enriched_doc: Dict[str, Any]) -> Dict[str, str]:
        """
        Asigna un 'line_id' a cada polígono y construye la geometría de las líneas.
        Devuelve un mapeo polygon_id -> line_id.
        """
        polygons = enriched_doc.get("polygons", {})
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
        
    def _get_lines_geometry(self) -> Dict[str, Any]:
        """
        Devuelve la geometría de las líneas construida.
        """
        return self.lines_info.copy()
