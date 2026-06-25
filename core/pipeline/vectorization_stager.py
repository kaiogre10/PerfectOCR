# PerfectOCR/coordinators/tensoring_coordinator.py
import logging
import time
from typing import Any, Dict, Optional
from domain.data_formatter import DataFormatter
from domain.abstract_stager import AbstractStager

logger = logging.getLogger(__name__)

class VectorizationStager(AbstractStager):
    """Inicializa el coordinador y sus workers."""
    def execute(self, manager: DataFormatter, context: Optional[Dict[str, Any]] = None) -> Optional[DataFormatter]:
        """Orquesta el flujo completo de vectorización"""
        try:
            exec_context: Dict[str, Any] = context if context else {}
            time_worker_log = exec_context.get("time_worker_log")

            for worker_idx, worker in enumerate(self.workers):
                worker_name = worker.__class__.__name__
                exec_context["worker_name"] = worker_name
                logger.debug(f"Iniciando: {worker_idx + 1}/{len(self.workers)}: {worker_name}")

                worker_start = time.perf_counter()
                if not worker.vectorize(exec_context, manager):
                    logger.error(f"'{worker_name}' falló, tiempo: {time.perf_counter() - worker_start:.6f}'s")
                    manager = None
                    return manager
                
                if time_worker_log:
                    logger.info(f"'{worker_name}' completado en: {time.perf_counter() - worker_start:.6f}'s")
            
            return manager
        except Exception as e:
            logger.error(f"Error en vectorización: '{e}'", exc_info=True)
            raise