# core/workers/image_preparation/geometry_detector.py
import os
import logging
import time
from typing import Dict, Any, List, Optional
import numpy as np
from paddleocr import PaddleOCR
from core.workers.factory.abstract_worker import AbstractWorker

logger = logging.getLogger(__name__)

class GeometryDetector(AbstractWorker):
    """
    Detecta geometría con PaddleOCR y escribe resultados en el dict job_data:
    - job_data['image_data']['polygons'][poly_id] = { polygon_id, geometry, ... }
    No usa WorkflowJob.
    """
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.paddle_config = config
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
                    "lang": self.paddle_config.get("lang", "es"),
                    "show_log": self.paddle_config.get("show_log", False),
                    "use_gpu": self.paddle_config.get("use_gpu", False),
                    "enable_mkldnn": self.paddle_config.get("enable_mkldnn", True),
                }
                det_model_path = self.paddle_config.get("det_model_dir", "")
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

    def process(self, image: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        """
        - Lee 'job_data' desde context
        - Detecta bounding polys
        - Escribe en job_data['image_data']['polygons']
        - Retorna la misma imagen (no la modifica)
        """
        dict_data: Optional[Dict[str, Any]] = context.get("dict_data")
        if not dict_data:
            logger.warning("GeometryDetector: No se encontró 'dict_data' en el contexto.")
            return image

        # Asegurar estructura mínima
        if "image_data" not in dict_data:
            dict_data["image_data"] = {}
        if "polygons" not in dict_data["image_data"]:
            dict_data["image_data"]["polygons"] = {}

        try:
            dict_data = {
                "metadata": dict_data.get("metadata", {}),
                "polygons": {},
            }
            dict_data = self._detect_dict(image, dict_data)

            # Persistir polígonos detectados en job_data
            polygons = dict_data.get("polygons", {})
            if not polygons:
                logger.warning("[GeometryDetector] No se encontraron polígonos")
                return image

            dict_polygons = dict_data["image_data"]["polygons"]
            added = 0
            for poly_id, poly_entry in polygons.items():
                # No forzamos schema completo aquí: solo geometría básica.
                # line_id, cropedd_geometry y ocr se llenan en etapas posteriores.
                dict_polygons[poly_id] = {
                    "polygon_id": str(poly_id),
                    "geometry": {
                        "polygon_coords": poly_entry["geometry"]["polygon_coords"],
                        "bounding_box": poly_entry["geometry"]["bounding_box"],
                        "centroid": poly_entry["geometry"]["centroid"],
                    },
                    # placeholders mínimos opcionales (no obligatorios aquí)
                    # "line_id": None,
                    # "cropped_img": None,
                    # "cropedd_geometry": {...},
                    # "ocr": {"ocr_raw": None, "confidence": 0.0}
                }
                added += 1

            logger.info(f"[GeometryDetector] Polígonos escritos en dict_data: {added}")
        except Exception as e:
            logger.error(f"Error en GeometryDetector: {e}", exc_info=True)

        return image  # No modificamos la imagen

    def _detect_dict(self, current_image: np.ndarray, dict_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta la detección (PaddleOCR det=False, rec=False) y devuelve:
        doc_data['polygons'] = { 'poly_0000': { 'polygon_id', 'geometry': {...} }, ... }
        """
        if self.engine is None:
            logger.error("GeometryDetector: motor PaddleOCR no inicializado.")
            dict_data["polygons"] = {}
            return dict_data

        try:
            results = self.engine.ocr(current_image, cls=False, rec=False)
            if not (results and len(results) > 0 and results[0] is not None):
                logger.warning("La detección geométrica no encontró polígonos de texto.")
                dict_data["polygons"] = {}
                return dict_data
        except Exception as e:
            logger.error(f"Error durante la detección con PaddleOCR: {e}", exc_info=True)
            dict_data["polygons"] = {}
            return dict_data

        polygons_dict: Dict[str, Any] = {}
        for idx, bbox_polygon_raw in enumerate(results[0]):
            # results[0] suele ser lista de polígonos, cada uno con 4 puntos [x,y]
            if not isinstance(bbox_polygon_raw, list) or len(bbox_polygon_raw) < 3:
                continue
            try:
                polygon_coords: List[List[float]] = [[float(p[0]), float(p[1])] for p in bbox_polygon_raw]
                xs = [pt[0] for pt in polygon_coords]
                ys = [pt[1] for pt in polygon_coords]

                poly_id = f"poly_{idx:04d}"
                polygons_dict[poly_id] = {
                    "polygon_id": poly_id,
                    "geometry": {
                        "polygon_coords": polygon_coords,
                        "bounding_box": [min(xs), min(ys), max(xs), max(ys)],
                        "centroid": [float(sum(xs)) / len(xs), float(sum(ys)) / len(ys)],
                    },
                }
            except Exception as e:
                logger.debug(f"Omitiendo polígono inválido idx={idx}: {e}")
                continue

        dict_data["polygons"] = polygons_dict
        return dict_data
