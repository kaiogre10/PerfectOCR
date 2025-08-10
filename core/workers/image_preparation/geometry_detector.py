# core/workers/image_preparation/geometry_detector.py
import os
import logging
import time
import numpy as np
from typing import Dict, Any, Optional, List
from paddleocr import PaddleOCR  # type: ignore
from core.factory.abstract_worker import AbstractWorker
from core.domain.data_formatter import DataFormatter
import cv2

logger = logging.getLogger(__name__)

class GeometryDetector(AbstractWorker):
    """
    Detecta geometría con PaddleOCR y escribe resultados en el dict job_data:
    - job_data['image_data']['polygons'][poly_id] = { polygon_id, geometry, ... }
    No usa WorkflowJob.
    """
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.config = config
        self.init_params = config.get("paddle_det_config", {})
        self._engine = None
            
    @property
    def engine(self) -> Optional[PaddleOCR]:
        if self._engine is None:
            start_time = time.perf_counter()
            try:
                # Configurar parámetros de inicialización
                init_params = {
                    "use_angle_cls": False,
                    "rec": False,
                    "lang": self.config.get("paddle_config", {}).get("lang", "es"),
                    "show_log": self.config.get("paddle_config", {}).get("show_log", False),
                    "use_gpu": self.config.get("paddle_config", {}).get("use_gpu", False),
                    "enable_mkldnn": self.config.get("paddle_config", {}).get("enable_mkldnn", True),
                    "det_model_dir": "C:/PerfectOCR/data/models/paddle/det/es"
                }
                
                # Verificar la ruta del modelo de detección
                det_model_path = init_params["det_model_dir"]
                if det_model_path:
                    if os.path.exists(det_model_path):
                        init_params["det_model_dir"] = det_model_path
                        logger.info(f"Usando modelo de detección en: {det_model_path}")
                    else:
                        logger.warning(f"Ruta del modelo de detección no válida: {det_model_path}")
                else:
                    logger.warning("No se especificó 'det_model_dir'; PaddleOCR intentará descargar el modelo.")

                # Inicializar el modelo
                load_t0 = time.perf_counter()
                logger.info("Inicializando motor PaddleOCR...")
                self._engine = PaddleOCR(**init_params)
                logger.info(
                    f"PaddleOCR (det) listo en {time.perf_counter()-start_time:.3f}s "
                    f"(carga modelo: {time.perf_counter()-load_t0:.3f}s)"
                )
            except Exception as e:
                logger.error(f"Error inicializando PaddleOCR para geometría: {e}", exc_info=True)
                self._engine = None
        
        # Verificación adicional para asegurar que el motor no sea None
        if self._engine is None:
            logger.error("Motor PaddleOCR no inicializado correctamente.")
        
        return self._engine

    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        img: Optional[np.ndarray[Any, Any]] = context.get("full_img")
        if img is None:
            logger.error("GeometryDetector: full_img no encontrado")
            return False

        # Verificar que el motor esté inicializado
        engine = self.engine
        if engine is None:
            logger.error("GeometryDetector: Motor PaddleOCR no inicializado")
            return False

        try:
            # TRUCO: Convertir de gris a BGR para engañar a PaddleOCR
            if len(img.shape) == 2:  # Si es escala de grises
                img_for_paddle = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                logger.info("GeometryDetector: Convirtiendo imagen de gris a BGR para PaddleOCR")
            else:
                img_for_paddle = img

            # Llamamos explícitamente a engine.ocr() después de verificar que engine no es None
            results:Optional[List[Any]] = engine.ocr(img_for_paddle, det=True, cls=False, rec=False)
            if not (results and len(results) > 0 and results[0] is not None):
                logger.warning("La detección geométrica no encontró polígonos de texto.")
                return False
                
            if manager.workflow is None:
                manager.create_dict(
                    dict_id=context.get("dict_id", "default"),
                    full_img=img,
                        metadata=context.get("metadata", {})
                    )
            success = manager.create_polygon_dicts(results)
            if not success:
                logger.error("GeometryDetector: Fallo al estructurar polígonos.")
                return False
            logger.info("GeometryDetector: Polígonos estructurados correctamente.")

        except Exception as e:
            logger.error(f"Error durante la detección con PaddleOCR: {e}", exc_info=True)
            return False