# PerfectOCR/core/pipeline/input_stager.py
import logging
import time
from typing import Optional, Tuple, List, Dict, Any
from core.workers.image_preparation.image_loader import ImageLoader
from core.workers.factory.abstract_worker import AbstractWorker

logger = logging.getLogger(__name__)

class InputStager:
    def __init__(self, workers_factory: List[AbstractWorker], image_loader: ImageLoader, project_root: str):
        self.project_root = project_root
        self.workers_factory = workers_factory
        self.image_loader = image_loader

    def generate_polygons(self) -> Tuple[Optional[Dict[str, Any]], float]:
        start_time = time.time()

        # 1) Cargar dict autosuficiente (job_data)
        dict_data = self.image_loader.load_image_and_metadata()
        if not dict_data or dict_data.get("full_img") is None:
            logger.error("No se pudo cargar la imagen o crear el dict inicial")
            return None, 0.0

        # 2) Contexto simple para los workers
        context = {
            "": dict_data,
            "metadata": dict_data.get("metadata", {})
            }

        current_image = dict_data["full_img"]

        # 3) Ejecutar workers en secuencia
        for worker in self.workers_factory:
            try:
                current_image = worker.process(current_image, context)
                dict_data["full_img"] = current_image
            except Exception as e:
                logger.error(f"Error en {worker.__class__.__name__}: {e}")
                return None, 0.0

        total_time = time.time() - start_time
        logger.info(f"[InputStager] Pipeline completado en: {total_time:.3f}s")

        return dict_data, total_time
