# core/workers/image_preparation/geometry_detector.py
import logging
import cv2
import time
import numpy as np
from typing import Dict, Any, Optional, List
from core.factory.abstract_worker import ImagePrepAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.utils.image_utils import binarice_img, normalice_image

logger = logging.getLogger(__name__)

class GeometryDetector(ImagePrepAbstractWorker):
    """
    Detecta geometría con PaddleOCR:
    """
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        worker_config = config.get('geometry_detector', {})
        self.min_area = worker_config.get("min_area")
        self.kernel_threshold = config["morph_kernel"]
        self.output = config.get("deleted_polys")
        self.output2 = config.get("opened")
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
            full_img = img_obj.full_img if img_obj is not None else None
                    
            if full_img is None:
                logger.error(f"No Hay full_img en el Formatter")
                return False

            logger.debug("Full_img obtenida con éxito")
            full_img = normalice_image(full_img)
            if full_img is None:
                return False
                
            bin_img = binarice_img(full_img, {})            
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
            img=cv2.morphologyEx(bin_img.copy(), cv2.MORPH_CLOSE, kernel, iterations=2)

            if self.output2:
                from services.output_service import save_croped_image
                worker_name = context.get("worker_name") or "geometry_detector"
                output_paths = context["output_paths"]
                image_name = manager.workflow.metadata.image_name if manager.workflow else ""
                # imag_id = f"opened_{image_name}_{worker_name}"
                img_id = f"bin_img_{image_name}_{worker_name}_+1"
                save_croped_image(image_name, img_id, img, output_paths, worker_name)
                # save_croped_image(image_name, imag_id, img, output_paths, worker_name)
                
            polygons: List[List[float]] = engine.ocr(img=img, det=True, cls=False, rec=False)

            if not (polygons and len(polygons) > 0 and polygons[0] is not None): # type: ignore
                logger.warning("GeometryDetector: No se encontraron polígonos de texto.")
                return False
            
            discarted_polys: List[str] = []
            final_polygons_list: List[Dict[str, Any]] = []
            
            for idx, poly_pts in enumerate(polygons[0]):
                poly_id = f"poly_{idx:04d}"
                poly_index = idx
                coords = np.array([[float(p[0]), float(p[1])] for p in poly_pts]) # type: ignore
                bbox = np.array([coords[:, 0].min(), coords[:, 1].min(), coords[:, 0].max(), coords[:, 1].max()])
                centroid = coords.mean(axis=0)

                # Calcular área para este bbox
                bbox_width = bbox[2] - bbox[0]
                bbox_height = bbox[3] - bbox[1]
                area = bbox_height * bbox_width

                if area < self.min_area:
                    logger.debug(f"Polígono {poly_id} descarcatdo por mínima área")
                    
                    if self.output:
                        from services.output_service import save_croped_image
                        from core.utils.image_utils import cropp_img
                        cropped = cropp_img(img, bbox)
                        worker_name = context.get("worker_name") or "geometry_detector"
                        output_paths = context["output_paths"]
                        pid = f"{poly_id}_{worker_name}"
                        image_name = manager.workflow.metadata.image_name if manager.workflow else ""
                        save_croped_image(image_name, pid, cropped, output_paths, worker_name)
                        
                    discarted_polys.append(poly_id)
                    continue

                final_polygons_list.append({
                    "poly_index": poly_index,
                    "polygon_coords": coords,
                    "bounding_box": bbox,
                    "centroid": centroid,
                })

            final_polygons: Dict[str, Dict[str, Any]] = {}
            for new_idx, poly_data in enumerate(final_polygons_list):
                poly_id = f"poly_{new_idx:04d}"
                poly_index = new_idx
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
