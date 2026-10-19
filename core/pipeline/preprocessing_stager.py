# PerfectOCR/core/coordinators/preprocessing_coordinator.py
import logging
import time
from typing import Any, Dict, Optional
from domain.data_formatter import DataFormatter
from core.contracts.abstract_stager import AbstractStager

logger = logging.getLogger(__name__)

class PreprocessingStager(AbstractStager):
    """Coordina la fase de preprocesamiento, delegando todo el trabajo a un único worker autosuficiente."""
    def execute(self, manager: DataFormatter, context: Optional[Dict[str, Any]] = None) -> Optional[DataFormatter]:
        time_worker_log = self.modules_config.get("time_worker_log")
        for worker_idx, worker in enumerate(self.workers):
            worker_name = worker.__class__.__name__

            logger.debug(f"Worker {worker_idx + 1}/{len(self.workers)}: {worker_name}")
            
            worker_start = time.perf_counter()
            if not worker.preprocess(context, manager):
                logger.error(f"'{worker_name}' falló, tiempo: {time.perf_counter() - worker_start:.6f}'s", exc_info=True)
                return None

            if time_worker_log:
                logger.info(f"'{worker_name}' completado en: {time.perf_counter() - worker_start:.6f}'s")

        return manager