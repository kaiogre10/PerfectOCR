#PerfectOCR/core/pipeline/input_stager.py
import logging
import time
from typing import Optional, Tuple, Dict, Any
from core.factory.abstract_stager import AbstractStager
from core.domain.data_formatter import DataFormatter

logger = logging.getLogger(__name__)

class ImagePreparationStager(AbstractStager):
    """Stager de preparación de imágenes."""
    
    def execute(self, manager: DataFormatter) -> Tuple[Optional[DataFormatter], float]:
        """Ejecuta la fase de preparación completa."""
        return self.prepare_image(manager)

    def prepare_image(self, manager: DataFormatter) -> Tuple[Optional[DataFormatter], float]:
        start_time = time.perf_counter()
        for worker_idx, worker in enumerate(self.workers):
            worker_start = time.perf_counter()
            worker_name = worker.__class__.__name__
            logger.debug(f"Ejecutando worker {worker_idx + 1}/{len(self.workers)}: {worker_name}")

            context: Dict[str, Any] = {
                "worker_name": worker_name,
                "output_paths": self.output_paths
            } 
            if not worker.process(context, manager):
                logger.error(f"Fallo en {worker.__class__.__name__}")
                return None, 0.0

            logger.info(f" {worker.__class__.__name__} completado en {time.perf_counter() - worker_start:.6f}s")
        
        total_time = time.perf_counter() - start_time
        logger.debug(f" Completado en {total_time:.6f}s")
        return manager, total_time