# PerfectOCR/core/preprocessing/poly_gone.py
import cv2
import numpy as np
import logging
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
        
        # Usar polígonos del WorkflowJob en lugar de doc_data
        if workflow_job and workflow_job.polygons:
            # Convertir objetos Polygon a formato dict para compatibilidad
            polygons_dict = {}
            for poly_id, polygon in workflow_job.polygons.items():
                polygons_dict[poly_id] = {
                    "polygon_id": polygon.polygon_id,
                    "geometry": {
                        "polygon_coords": polygon.geometry.polygon_coords,
                        "bounding_box": [polygon.geometry.bounding_box.x_min, polygon.geometry.bounding_box.y_min, 
                                       polygon.geometry.bounding_box.x_max, polygon.geometry.bounding_box.y_max],
                        "centroid": polygon.geometry.centroid
                    }
                }
            
            doc_data = {
                "metadata": metadata,
                "polygons": polygons_dict
            }
        else:
            doc_data = {
                "metadata": metadata,
                "polygons": {}
            }
        
        # Extraer polígonos individuales
        polygons = self._extract_individual_polygons(image, doc_data)
        
        # Actualizar el WorkflowJob con las imágenes recortadas
        if workflow_job and workflow_job.full_img is not None:
            # Actualizar los polígonos del WorkflowJob con las imágenes recortadas
            for poly_id, polygon in workflow_job.polygons.items():
                if poly_id in self.polygons_info:
                    polygon.cropped_img = self.polygons_info[poly_id]["cropped_img"]
                    polygon.padding_coords = self.polygons_info[poly_id]["padding_coords"]
            
            workflow_job.update_stage(ProcessingStage.POLYGONS_EXTRACTED)
        
        return image  # Retorna la misma imagen (no la modifica)
        
    def _extract_individual_polygons(self, image: np.ndarray[Any, Any], doc_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extrae y recorta los polígonos individuales."""
        polygons = doc_data.get("polygons", {})
        if not polygons:
            logger.warning("No se encontraron polígonos en el documento enriquecido")
            return {}
        
        padding = self.config.get('cropping_padding', 5)
        metadata = doc_data.get("metadata", {})
        img_dims = metadata.get("img_dims", {})
        
        # img_dims es un objeto ImageDimensions, no un diccionario
        if hasattr(img_dims, 'height') and hasattr(img_dims, 'width'):
            img_h = img_dims.height
            img_w = img_dims.width
        else:
            # Fallback si no es un objeto ImageDimensions
            img_h = int(img_dims.get("height", 0) if isinstance(img_dims, dict) else 0)
            img_w = int(img_dims.get("width", 0) if isinstance(img_dims, dict) else 0)

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
                    cropped_poly = image[poly_y1:poly_y2, poly_x1:poly_x2]
                    if cropped_poly.size > 0:
                        poly_data["cropped_img"] = cropped_poly
                        poly_data["padding_coords"] = [poly_x1, poly_y1, poly_x2, poly_y2]
                        self.polygons_info[poly_id] = {
                            "cropped_img": cropped_poly.copy(),
                            "padding_coords": [poly_x1, poly_y1, poly_x2, poly_y2]
                        }
                
            except Exception as e:
                logger.error(f"Error recortando polígono {poly_id}: {e}")
                poly_data["cropped_img"] = None

        return polygons
    
    def _get_polygons_copy(self) -> Dict[str, Any]:
        """
        Devuelve una copia de los polígonos con sus imágenes recortadas.
        """
        return self.polygons_info.copy()
