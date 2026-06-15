# PerfectOCR/core/coordinators/ocr_manager.py
import time
import logging
from typing import Optional, Dict, Any, Tuple
from core.domain.data_formatter import DataFormatter
from core.factory.abstract_stager import AbstractStager

logger = logging.getLogger(__name__)

class OCRStager(AbstractStager):
    """Stager ee reconocimiento óptico de caracteres."""
    
    def execute(self, manager: DataFormatter, context: Optional[Dict[str, Any]] = None) -> Tuple[Optional[DataFormatter], float]:
        """Ejecuta la fase de OCR completa."""
        return self.run_ocr_on_polygons(manager, context)
        
    def run_ocr_on_polygons(self, manager: DataFormatter, context: Optional[Dict[str, Any]] = None) -> Tuple[Optional[DataFormatter], float]:
        start_time = time.perf_counter()
        exec_context: Dict[str, Any] = context.copy() if context else {}
        time_worker_log = exec_context.get("time_worker_log")

        for worker_idx, worker in enumerate(self.workers):
            worker_name = worker.__class__.__name__
            logger.debug(f"Inicia Worker: {worker_idx + 1}/{len(self.workers)}: {worker_name}")

            exec_context["worker_name"] = worker_name  # Actualiza el nombre en cada iteración

            worker_start = time.perf_counter()
            if not worker.transcribe(exec_context, manager):
                worker_time = time.perf_counter() - start_time
                logger.error(f"'{worker_name}' falló, tiempo: {worker_time:.6f}'s", exc_info=True)
                return None, worker_time
            
            if time_worker_log:
                logger.info(f"'{worker_name} completado en: {time.perf_counter() - worker_start:.6f}s")

        return manager, time.perf_counter() - start_time