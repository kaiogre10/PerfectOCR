# PerfectOCR/core/pipeline/input_stager.py
import logging
import time
from typing import Optional, Tuple, List, Dict, Any
from core.domain.data_formatter import DataFormatter
from core.factory.abstract_worker import ImagePrepAbstractWorker

logger = logging.getLogger(__name__)

class ImagePreparationStager:
    def __init__(self, workers: List[ImagePrepAbstractWorker], stage_config: Dict[str, Any], output_paths: Optional[List[str]], project_root: str):
        self.project_root = project_root
        self.workers = workers
        self.stage_config = stage_config
        self.output_paths = output_paths

    def generate_polygons(self, manager: DataFormatter) -> Tuple[Optional[DataFormatter], float]:
        start_time = time.time()

        # 3) Contexto con metadatos necesarios
        context: Dict[str, Any] = {
            "output_paths": self.output_paths,
            "project_root": self.project_root
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
        logger.debug(f"[InputStager] Módulo 1 completado en: {total_time:.3f}s")
        return manager, total_time
