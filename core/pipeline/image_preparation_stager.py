# PerfectOCR/core/pipeline/input_stager.py
import logging
import time
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from core.domain.data_formatter import DataFormatter
from core.workers.image_preparation.image_loader import ImageLoader
from core.factory.abstract_worker import ImagePrepAbstractWorker

logger = logging.getLogger(__name__)

class ImagePreparationStager:
    def __init__(self, workers: List[ImagePrepAbstractWorker], image_loader: ImageLoader, project_root: str):
        self.project_root = project_root
        self.workers = workers
        self._image_loader = image_loader

    def generate_polygons(self, manager: DataFormatter) -> Tuple[Optional[DataFormatter], float]:
        start_time = time.time()
        # 1) Cargar datos crudos (sin manager)
        gray_image, metadata = self._image_loader.load_image_and_metadata()
        if gray_image is None or not isinstance(gray_image, np.ndarray) or gray_image.size == 0:
            logger.error(f"InputStager: Imagen no válida para '{metadata.get('image_name')}")
            return None, 0.0
        
        # 2) Crear manager y dict una sola vez
        manager = DataFormatter()
        dict_id = f"dict_{metadata.get('image_name')}_{int(time.time())}"

        full_img = gray_image
        if not manager.create_dict(dict_id, full_img, metadata):
            logger.error("InputStager: Fallo al crear dict_job en el manager.")
            return None, 0.0

        # 3) Contexto con metadatos necesarios
        context: Dict[str, Any] = {
            "dict_id": dict_id,
            "full_img": full_img,
            "metadata": metadata,
            "img_dims": metadata.get("img_dims", {}),
        }
        # 4) Ejecutar workers (inyectar context y manager) y loguear tiempo de cada uno
        for worker in self.workers:
            worker_start = time.time()
            if not worker.process(context, manager):
                logger.error(f"InputStager: Fallo en el worker {worker.__class__.__name__}")
                return None, 0.0
            worker_time = time.time() - worker_start
            logger.debug(f"[InputStager] Worker {worker.__class__.__name__} completado en: {worker_time:.3f}s")

        total_time = time.time() - start_time
        logger.info(f"[InputStager] Módulo 1 completado en: {total_time:.3f}s")
        return manager, total_time
