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
        exec_context: Dict[str, Any] = context if context else {}

        time_worker_log = exec_context.get("time_worker_log")

        for worker_idx, worker in enumerate(self.workers):
            worker_name = worker.__class__.__name__
            logger.debug(f"Ejecutando {worker_idx + 1}/{len(self.workers)}: '{worker_name}'")

            exec_context["worker_name"] = worker_name  # Actualiza el nombre en cada iteración
            
            worker_start = time.perf_counter()
            if not worker.process(exec_context, manager):
                worker_time = time.perf_counter() - start_time
                logger.error(f"'{worker_name}' falló, tiempo: {worker_start:.6f}'s", exc_info=True)
                return None, worker_time

            if time_worker_log:
                logger.info(f"'{worker_name}' completado en: {time.perf_counter() - worker_start:.6f}'s")

        return manager, time.perf_counter() - start_time