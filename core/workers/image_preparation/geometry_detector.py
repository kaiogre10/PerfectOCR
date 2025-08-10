# core/workers/image_preparation/geometry_detector.py
import os
import logging
import time
from typing import Dict, Any, Optional
from paddleocr import PaddleOCR
from core.factory.abstract_worker import AbstractWorker
from core.domain.data_manager import DataFormatter

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
        self._engine = None
        logger.info("GeometryDetector inicializado (motor PaddleOCR no cargado aún).")

    @property
    def engine(self) -> Optional[PaddleOCR]:
        if self._engine is None:
            start_time = time.perf_counter()
            try:
                init_params = {
                    "use_angle_cls": False,
                    "rec": False,
                    "lang": self.config.get("lang", "es"),
                    "show_log": self.config.get("show_log", False),
                    "use_gpu": self.config.get("use_gpu", False),
                    "enable_mkldnn": self.config.get("enable_mkldnn", True),
                }
                det_model_path = self.config.get("det_model_dir", "")
                if det_model_path and os.path.exists(det_model_path):
                    init_params["det_model_dir"] = det_model_path
                else:
                    logger.warning("No se encontró 'det_model_dir'; PaddleOCR intentará descargar el modelo.")

                load_t0 = time.perf_counter()
                self._engine = PaddleOCR(**init_params)
                logger.info(
                    f"PaddleOCR (det) listo en {time.perf_counter()-start_time:.3f}s "
                    f"(carga modelo: {time.perf_counter()-load_t0:.3f}s)"
                )
            except Exception as e:
                logger.error(f"Error inicializando PaddleOCR para geometría: {e}", exc_info=True)
                self._engine = None
        return self._engine

    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """
        - Lee 'job_data' desde context
        - Detecta bounding polys
        - Escribe en job_data['image_data']['polygons']
        - Retorna la misma imagen (no la modifica)
        """
        img = context.get("full_img")
        if img is None:
            logger.error("GeometryDetector: full_img no encontrado")
            return False

        try:
            results = self.engine.ocr(img, cls=False, rec=False)
            if not (results and len(results) > 0 and results[0] is not None):
                logger.warning("La detección geométrica no encontró polígonos de texto.")
                return True
        except Exception as e:
            logger.error(f"Error durante la detección con PaddleOCR: {e}", exc_info=True)
            return False

        abstract: Dict[str, Dict[str, Any]] = {}
        for idx, poly_pts in enumerate(results[0]):
            xs = [float(p[0]) for p in poly_pts]
            ys = [float(p[1]) for p in poly_pts]
            poly_id = f"poly_{idx:04d}"
            abstract.setdefault("polygon_id", {})[poly_id] = poly_id
            abstract.setdefault("polygon_coords", {})[poly_id] = [[xs[i], ys[i]] for i in range(len(xs))]
            abstract.setdefault("bounding_box", {})[poly_id] = [min(xs), min(ys), max(xs), max(ys)]
            abstract.setdefault("centroid", {})[poly_id] = [sum(xs) / len(xs), sum(ys) / len(ys)]

        if abstract:
            ok = manager.update_data(abstract)
            if not ok:
                logger.error("GeometryDetector: fallo al escribir polígonos")
                return False
        return True
