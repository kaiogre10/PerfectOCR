#PerfectOCR/core/pipeline/input_stager.py
import logging
import time
from typing import Optional, Tuple, Dict, Any
from core.factory.abstract_stager import AbstractStager
from core.domain.data_formatter import DataFormatter

logger = logging.getLogger(__name__)

class ImagePreparationStager(AbstractStager):
    """Stager de preparación de imágenes."""
    
    def execute(self, manager: DataFormatter, context: Optional[Dict[str, Any]] = None) -> Tuple[Optional[DataFormatter], float]:
        """Ejecuta la fase de preparación completa."""
        return self.prepare_image(manager, context)

    def prepare_image(self, manager: DataFormatter, context: Optional[Dict[str, Any]] = None) -> Tuple[Optional[DataFormatter], float]:
        start_time = time.perf_counter()
        
        # Usar contexto base si existe, sino crear uno nuevo
        exec_context: Dict[str, Any] = context.copy() if context else {}
        
        # Asegurar output_paths en el contexto
        if "output_paths" not in exec_context:
            exec_context["output_paths"] = self.output_paths

        for worker_idx, worker in enumerate(self.workers):
            worker_start = time.perf_counter()
            worker_name = worker.__class__.__name__
            logger.debug(f"Ejecutando worker {worker_idx + 1}/{len(self.workers)}: {worker_name}")

            exec_context["worker_name"] = worker_name  # Actualiza el nombre en cada iteración

            if not worker.process(exec_context, manager):
                logger.error(f"Fallo en {worker.__class__.__name__}", exc_info=True)
                return None, 0.0
            
            if manager.workflow:
                worker_time = time.time() - worker_start
                logger.debug(f"Worker {worker_name} completado en: {worker_time:.6f}s")

            logger.debug(f" {worker.__class__.__name__} completado en {time.perf_counter() - worker_start:.6f}s")
        
        total_time = time.perf_counter() - start_time
        logger.debug(f" Completado en {total_time:.6f}s")
        return manager, total_time