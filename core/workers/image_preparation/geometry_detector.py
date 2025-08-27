# core/workers/image_preparation/geometry_detector.py
import logging
import numpy as np
from typing import Dict, Any, Optional, List
from core.factory.abstract_worker import ImagePrepAbstractWorker
from core.domain.data_formatter import DataFormatter

logger = logging.getLogger(__name__)

class GeometryDetector(ImagePrepAbstractWorker):
    """
    Detecta geometría con PaddleOCR y escribe resultados en el workflow_dict:
    workflow_dict['polygons'][poly_id] = { polygon_id, geometry, ... }
    No usa WorkflowJob.
    """
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.config = config
        self._engine = None
            
    @property
    def engine(self) -> Optional[Any]:
        if self._engine is None:
            from core.domain.models_manager import PaddleManager
            paddle_manager = PaddleManager.get_instance()
            self._engine = paddle_manager.detection_engine
            
            if self._engine is None:
                logger.error("GeometryDetector: Motor de detección no disponible en PaddleManager")
            else:
                logger.debug("GeometryDetector: Motor de detección obtenido del PaddleManager")
        
        return self._engine
        
    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        try:
            img: Optional[np.ndarray[Any, np.dtype[np.uint8]]] = context.get("full_img")
            if img is None:
                logger.error("GeometryDetector: full_img no encontrado en el contexto.")
                return False

            engine = self.engine
            if engine is None:
                logger.error("GeometryDetector: Motor PaddleOCR no inicializado.")
                return False

            results: Optional[List[Any]] = engine.ocr(img=img, det=True, cls=False, rec=False) 
            logger.debug(f"GeometryDetector: Resultados de OCR obtenidos: {len(results[0]) if results and results[0] is not None else 0} polígonos.")

            if not (results and len(results) > 0 and results[0] is not None):
                logger.warning("GeometryDetector: No se encontraron polígonos de texto.")
                return False

            success = manager.create_polygon_dicts(results)
            if not success:
                logger.error("GeometryDetector: Fallo al estructurar polígonos.")
                return False

            return True
        
        except Exception as e:
            logger.error(f"Error en procesamiento vectorizado de geometría: {e}")
            return False
