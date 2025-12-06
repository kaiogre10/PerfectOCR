# core/workers/image_preparation/geometry_detector.py
import logging
import time
import numpy as np
from typing import Dict, Any, Optional, List
from core.factory.abstract_worker import ImagePrepAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.utils.image_utils import validate_image

logger = logging.getLogger(__name__)

class GeometryDetector(ImagePrepAbstractWorker):
    """
    Detecta geometría con PaddleOCR:
    """
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get('geometry_detector', {})
        self.min_area = self.worker_config.get("min_area")
        self.output = config.get("deleted_polys", False)
        self._engine = None
            
    @property
    def engine(self) -> Optional[Any]:
        if self._engine is None:
            from core.domain.models_manager import ModelsManager
            paddle_manager = ModelsManager.get_instance()
            self._engine = paddle_manager.detection_engine            
            if self._engine is None:
                logger.error("GeometryDetector: Motor de detección no disponible en PaddleManager")
        
            logger.debug("GeometryDetector: Motor de detección obtenido del PaddleManager")
    
        return self._engine
        
    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        start_time = time.perf_counter()
        try:
            engine = self.engine
            if engine is None:
                logger.error("PaddleOCR no inicializado.")
                return False

            img_obj = manager.get_full_img()
            img = img_obj.full_img if img_obj is not None else None
            if not validate_image(img):
            # if img is None:
                logger.error(f"No Hay full_img en el Formatter")
                return False

            logger.debug("Full_img obtenida con éxito")

            polygons: List[List[float]] = engine.ocr(img=img, det=True, cls=False, rec=False)

            if not (polygons and len(polygons) > 0 and polygons[0] is not None): # type: ignore
                logger.warning("GeometryDetector: No se encontraron polígonos de texto.")
                return False
            
            discarted_polys: List[str] = []
            final_polygons_list: List[Dict[str, Any]] = []
            
            for idx, poly_pts in enumerate(polygons[0]):
                poly_id = f"poly_{idx:04d}"
                coords = np.array([[float(p[0]), float(p[1])] for p in poly_pts]) # type: ignore
                bbox = np.array([coords[:, 0].min(), coords[:, 1].min(), coords[:, 0].max(), coords[:, 1].max()])
                centroid = coords.mean(axis=0)

                # Calcular área para este bbox
                bbox_width = bbox[2] - bbox[0]
                bbox_height = bbox[3] - bbox[1]
                area = bbox_height * bbox_width

                if area < self.min_area:
                    # logger.info(f"Polígono {poly_id} descarcatdo por mínima área")
                    
                    if self.output:
                        from services.output_service import save_croped_image
                        from core.utils.image_utils import cropp_img
                        cropped = cropp_img(img, bbox) # type: ignore
                        worker_name = context.get("worker_name") or "geometry_detector"
                        output_paths = context["output_paths"]
                        pid = f"{poly_id}_{worker_name}"
                        image_name = manager.workflow.metadata.image_name if manager.workflow else ""
                        save_croped_image(image_name, pid, cropped, output_paths, worker_name) # type: ignore
                        
                    discarted_polys.append(poly_id)
                    continue

                final_polygons_list.append({
                    "polygon_coords": coords,
                    "bounding_box": bbox,
                    "centroid": centroid,
                })

            final_polygons: Dict[str, Dict[str, Any]] = {}
            for new_idx, poly_data in enumerate(final_polygons_list):
                poly_id = f"poly_{new_idx:04d}"
                final_polygons[poly_id] = poly_data

            # logger.info(f"FINAL: {final_polygons}")
            logger.info(f"Polígonos inciales: {len(polygons[0])}, finales: {len(final_polygons)}, descartados {len(discarted_polys)}: {discarted_polys}")

            if not manager.create_polygon_dicts(final_polygons):
                logger.error("GeometryDetector: Fallo al estructurar polígonos.")
                return False

            else:
                logger.debug(f"{len(final_polygons)} poligonos válidos detectados en: {time.perf_counter()-start_time:.6f}s")
                return True
        
        except Exception as e:
            logger.error(f"Error en procesamiento vectorizado de geometría: {e}", exc_info=True)
            return False
