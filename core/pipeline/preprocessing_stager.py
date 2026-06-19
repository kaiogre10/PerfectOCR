# PerfectOCR/core/coordinators/preprocessing_coordinator.py
import logging
import time
from typing import Any, Dict, Tuple, Optional
from core.domain.data_formatter import DataFormatter
from core.factory.abstract_stager import AbstractStager

logger = logging.getLogger(__name__)

class PreprocessingStager(AbstractStager):
    """Coordina la fase de preprocesamiento, delegando todo el trabajo a un único worker autosuficiente."""
    def excute(self, manager: DataFormatter, context: Optional[Dict[str, Any]] = None) -> Tuple[Optional[DataFormatter], float]:
        start_time = time.perf_counter()
        
        # Base context setup
        exec_context: Dict[str, Any] = context.copy() if context else {}
        time_worker_log = exec_context.get("time_worker_log")

        for worker_idx, worker in enumerate(self.workers):
            worker_name = worker.__class__.__name__
            exec_context["worker_name"] = worker_name

            logger.debug(f"Worker {worker_idx + 1}/{len(self.workers)}: {worker_name}")
            
            worker_start = time.perf_counter()
            if not worker.preprocess(exec_context, manager):
                worker_time = time.perf_counter() - start_time
                logger.error(f"'{worker_name}' falló, tiempo: {worker_start:.6f}'s", exc_info=True)
                return None, worker_time

            if time_worker_log:
                logger.info(f"'{worker_name}' completado en: {time.perf_counter() - worker_start:.6f}'s")

        return manager, time.perf_counter() - start_time