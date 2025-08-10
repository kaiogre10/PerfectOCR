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
        logger.info("GeometryDetector: Iniciando proceso de detección geométrica.")
        img: Optional[np.ndarray[Any, Any]] = context.get("full_img")
        if img is None:
            logger.error("GeometryDetector: full_img no encontrado en el contexto.")
            return False

        engine = self.engine
        if engine is None:
            logger.error("GeometryDetector: Motor PaddleOCR no inicializado.")
            return False

        try:
            logger.info(f"GeometryDetector: Imagen recibida con shape {img.shape}.")
            if len(img.shape) == 2:
                img_for_paddle = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                logger.info("GeometryDetector: Convertida imagen de gris a BGR para PaddleOCR.")
            else:
                img_for_paddle = img
                logger.info("GeometryDetector: Imagen ya está en formato BGR.")

            logger.info("GeometryDetector: Ejecutando engine.ocr()...")
            results: Optional[List[Any]] = engine.ocr(img_for_paddle, det=True, cls=False, rec=False) # type: ignore
            logger.info(f"GeometryDetector: Resultados de OCR obtenidos: {len(results[0]) if results and results[0] is not None else 0} polígonos.")

            if not (results and len(results) > 0 and results[0] is not None):
                logger.warning("GeometryDetector: No se encontraron polígonos de texto.")
                return False

            if manager.workflow is None:
                logger.info("GeometryDetector: Workflow no inicializado, creando dict.")
                manager.create_dict(
                    dict_id=context.get("dict_id", "default"),
                    full_img=img,
                    metadata=context.get("metadata", {})
                )
            else:
                logger.info("GeometryDetector: Workflow ya inicializado.")

            logger.info("GeometryDetector: Estructurando polígonos en el manager...")
            success = manager.create_polygon_dicts(results)
            if not success:
                logger.error("GeometryDetector: Fallo al estructurar polígonos.")
                return False
            logger.info("GeometryDetector: Polígonos estructurados correctamente.")

        except Exception as e:
            logger.error(f"GeometryDetector: Error durante la detección con PaddleOCR: {e}", exc_info=True)
            return False

        logger.info("GeometryDetector: Proceso finalizado correctamente.")
        return True