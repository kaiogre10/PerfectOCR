# PerfectOCR/core/pipeline/input_stager.py
import logging
import time
from typing import Optional, Tuple, List, Dict, Any, cast
import numpy as np
from core.domain.data_manager import DataFormatter
from core.workers.image_preparation.image_loader import ImageLoader
from core.factory.abstract_worker import AbstractWorker

logger = logging.getLogger(__name__)

class InputStager:
    def __init__(self, workers: List[AbstractWorker], image_loader: ImageLoader, project_root: str):
        self.project_root = project_root
        self.workers = workers
        self._image_loader = image_loader

    def generate_polygons(self, manager: DataFormatter) -> Tuple[Optional[DataFormatter], float]:
        start_time = time.time()

        # 1) Cargar datos crudos (sin manager)
        gray_image, metadata = self._image_loader.load_image_and_metadata()
        if gray_image is None:
            logger.error("InputStager: No se pudo cargar la imagen.")

        # 2) Crear manager y dict_job una sola vez aquí
        manager = DataFormatter()
        dict_id = f"dict_{metadata.get('image_name')}_{int(time.time())}"

        # Cast para callar warning de tipo parcialmente desconocido (sin # type: ignore)
        full_img = cast(np.ndarray[Any, Any], gray_image)
        if not manager.create_dict(dict_id, full_img, metadata):
            logger.error("InputStager: Fallo al crear dict_job en el manager.")
            return None, 0.0

        # 3) Contexto liviano
        context: Dict[str, Any] = {
            "dict_id": dict_id,
            "full_img": full_img,
        }

        # 4) Ejecutar workers (inyectar context y manager)
        for worker in self.workers:
            if not worker.process(context, manager):
                logger.error(f"InputStager: Fallo en el worker {worker.__class__.__name__}")
                return None, 0.0

        total_time = time.time() - start_time
        logger.info(f"[InputStager] Pipeline completado en: {total_time:.3f}s")
        return manager, total_time
