# PerfectOCR/coordinators/tensoring_coordinator.py
import logging
import time
from typing import Any, Dict, Tuple, Optional
from core.domain.data_formatter import DataFormatter
from core.factory.abstract_stager import AbstractStager

logger = logging.getLogger(__name__)

class VectorizationStager(AbstractStager):
    """Inicializa el coordinador y sus workers. """    
    def execute(self, manager: DataFormatter, context: Optional[Dict[str, Any]] = None) -> Tuple[Optional[DataFormatter], float]:
        """Orquesta el flujo completo de vectorización"""
        start_time = time.perf_counter()
        try:
            exec_context: Dict[str, Any] = context.copy() if context else {}
            time_worker_log = exec_context.get("time_worker_log")

            for worker_idx, worker in enumerate(self.workers):
                worker_name = worker.__class__.__name__
                logger.debug(f"Iniciando: {worker_idx + 1}/{len(self.workers)}: {worker_name}")

                exec_context["worker_name"] = worker_name  # Actualiza el nombre en cada iteración

                worker_start = time.perf_counter()
                if not worker.vectorize(exec_context, manager):
                    worker_time = time.perf_counter() - worker_start
                    #logger.error(f"'{worker_name}' falló, tiempo: {worker_time:.6f}'s")
                    return None, worker_time
                
                if time_worker_log:
                    logger.info(f"'{worker_name}' completado en: {time.perf_counter() - worker_start:.6f}'s")
            
            return manager, time.perf_counter() - start_time
        except Exception as e:
            logger.error(f"Error en vectorización: '{e}'", exc_info=True)
        return None, time.perf_counter() - start_time