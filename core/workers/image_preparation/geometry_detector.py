# core/workers/image_preparation/geometry_detector.py
import logging
import numpy as np
from typing import Dict, Any, Optional, List
from core.factory.abstract_worker import ImagePrepAbstractWorker
from core.domain.data_formatter import DataFormatter
import cv2

logger = logging.getLogger(__name__)

class GeometryDetector(ImagePrepAbstractWorker):
    """
    Detecta geometría con PaddleOCR y escribe resultados en el workflow_dict:
    workflow_dict['polygons'][poly_id] = { polygon_id, geometry, ... }
    No usa WorkflowJob.
    """
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.config = config
        self._engine = None
            
    @property
    def engine(self) -> Optional[Any]:
        if self._engine is None:
            from core.domain.ocr_motor_manager import PaddleManager
            paddle_manager = PaddleManager.get_instance()
            self._engine = paddle_manager.detection_engine
            
            if self._engine is None:
                logger.error("GeometryDetector: Motor de detección no disponible en PaddleManager")
            else:
                logger.debug("GeometryDetector: Motor de detección obtenido del PaddleManager")
        
        return self._engine
        
    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        img: Optional[np.ndarray[Any, Any]] = context.get("full_img")
        if img is None:
            logger.error("GeometryDetector: full_img no encontrado en el contexto.")
            return False

        engine = self.engine
        if engine is None:
            logger.error("GeometryDetector: Motor PaddleOCR no inicializado.")
            return False

        try:
            if len(img.shape) == 2:
                img_for_paddle = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img_for_paddle = img

            results: Optional[List[Any]] = engine.ocr(img=img_for_paddle, det=True, cls=False, rec=False) 
            logger.debug(f"GeometryDetector: Resultados de OCR obtenidos: {len(results[0]) if results and results[0] is not None else 0} polígonos.")

            if not (results and len(results) > 0 and results[0] is not None):
                logger.warning("GeometryDetector: No se encontraron polígonos de texto.")
                return False

            if manager.workflow is None:
                manager.create_dict(
                    dict_id=context.get("dict_id", "default"),
                    full_img=img,
                    metadata=context.get("metadata", {})
                )
                
            else:
                logger.debug("GeometryDetector: Workflow ya inicializado.")

            success = manager.create_polygon_dicts(results)
            if not success:
                logger.error("GeometryDetector: Fallo al estructurar polígonos.")
                return False

        except Exception as e:
            logger.error(f"GeometryDetector: Error durante la detección con PaddleOCR: {e}", exc_info=True)
            return False

        return True