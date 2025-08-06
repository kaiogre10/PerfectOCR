# PerfectOCR/core/preprocessing/poly_gone.py
import cv2
import numpy as np
import logging
import os
from typing import Tuple, List, Dict, Any
from core.workers.factory.abstract_worker import AbstractWorker
from core.domain.workflow_job import ProcessingStage

logger = logging.getLogger(__name__)

class PolygonExtractor(AbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.config = config
        self.polygons_info: Dict[str, Any] = {}
        
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
        
        # Extraer polígonos individuales
        polygons = self._extract_individual_polygons(image, doc_data)
        
        # Actualizar el WorkflowJob si está disponible
        if workflow_job and workflow_job.full_img is not None:
            workflow_job.update_stage(ProcessingStage.POLYGONS_EXTRACTED)
        
        return image  # Retorna la misma imagen (no la modifica)
        
    def _extract_individual_polygons(self, deskewed_img: np.ndarray[Any, Any], enriched_doc: Dict[str, Any]) -> Dict[str, Any]:
        """Extrae y recorta los polígonos individuales."""
        polygons = enriched_doc.get("polygons", {})
        if not polygons:
            logger.warning("No se encontraron polígonos en el documento enriquecido")
            return {}
        
        padding = self.config.get('cropping_padding', 5)
        metadata = enriched_doc.get("metadata", {})
        img_dims = metadata.get("img_dims", {})
        img_h = int(img_dims.get("height", 0) or 0)
        img_w = int(img_dims.get("width", 0) or 0)

        self.polygons_info = {}
        
        for poly_id, poly_data in polygons.items():
            try:
                bbox = poly_data.get("geometry", {}).get("bounding_box")
                if not bbox:
                    continue

                poly_min_x, poly_min_y, poly_max_x, poly_max_y = map(int, bbox)
                poly_x1 = max(0, poly_min_x - padding)
                poly_y1 = max(0, poly_min_y - padding)
                poly_x2 = min(img_w, poly_max_x + padding)
                poly_y2 = min(img_h, poly_max_y + padding)

                if poly_x2 > poly_x1 and poly_y2 > poly_y1:
                    cropped_poly = deskewed_img[poly_y1:poly_y2, poly_x1:poly_x2]
                    if cropped_poly.size > 0:
                        poly_data["cropped_img"] = cropped_poly
                        poly_data["padding_coords"] = [poly_x1, poly_y1, poly_x2, poly_y2]
                        self.polygons_info[poly_id] = {"cropped_img": cropped_poly.copy()}
                
            except Exception as e:
                logger.error(f"Error recortando polígono {poly_id}: {e}")
                poly_data["cropped_img"] = None

        return polygons
    
    def _get_polygons_copy(self) -> Dict[str, Any]:
        """
        Devuelve una copia de los polígonos con sus imágenes recortadas.
        """
        return self.polygons_info.copy()
