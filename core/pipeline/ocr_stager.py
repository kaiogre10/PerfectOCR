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
        start_time = time.time()
        
        exec_context: Dict[str, Any] = context.copy() if context else {}
        if "output_paths" not in exec_context:
            exec_context["output_paths"] = self.output_paths

        for worker_idx, worker in enumerate(self.workers):
            worker_start = time.time()
            worker_name = worker.__class__.__name__
            logger.info(f"Inicia Worker: {worker_idx + 1}/{len(self.workers)}: {worker_name}")

            exec_context["worker_name"] = worker_name  # Actualiza el nombre en cada iteración

            if not worker.transcribe(exec_context, manager):
                logger.error(f"Worker {worker_name} falló o devolvió resultados vacíos", exc_info=True)
                return None, 0.0
            
            if manager.workflow:
                worker_time = time.time() - worker_start
                logger.info(f"Worker {worker_name} completado en: {worker_time:.6f}s")

        vect_time = time.time() - start_time
        logger.debug(f"Etapa 4 completado en: {vect_time:.6f}s")
        return manager, vect_time