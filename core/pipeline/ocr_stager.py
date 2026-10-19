# PerfectOCR/core/coordinators/ocr_manager.py
import time
import logging
from typing import Optional, Dict, Any, List
from domain.data_formatter import DataFormatter
from core.contracts.abstract_stager import AbstractStager

logger = logging.getLogger(__name__)

class OCRStager(AbstractStager):
    """FASE DE OCR Y POSTPROCESAMIENTO TEXTUAL"""
    __slots__ = ("time_worker_log", "workers")
    def __init__(self, workers: List[Any], config: Dict[str, Any]):
        super().__init__(workers, config)
        self.workers = workers
        self.time_worker_log = config.get("time_worker_log")

    def execute(self, manager: DataFormatter, context: Optional[Dict[str, Any]] = None) -> Optional[DataFormatter]:
        try:
            for worker_idx, worker in enumerate(self.workers):
                worker_name = worker.__class__.__name__
                logger.debug(f"Inicia Worker: {worker_idx + 1}/{len(self.workers)}: {worker_name}")
                worker_start = time.perf_counter()
                if not worker.transcribe(context, manager):
                    logger.error(f"'{worker_name}' falló, tiempo: {time.perf_counter() - worker_start:.6f}'s", exc_info=True)
                    return None

                if self.time_worker_log:
                    logger.info(f"'{worker_name}' completado en: {time.perf_counter() - worker_start:.6f}s")
                    continue

            return manager
        except Exception as e:
            logger.error(f"ERROR EN EL PROCESAMIENTO DE TEXTO: {e}", exc_info=True)
        return None