# PerfectOCR/core/coordinators/ocr_manager.py
import time
import logging
from typing import Optional, Dict, Any, Tuple
from core.domain.data_formatter import DataFormatter
from core.factory.abstract_stager import AbstractStager

logger = logging.getLogger(__name__)

class OCRStager(AbstractStager):
    """Stager de reconocimiento óptico de caracteres."""

    @property
    def config(self):
        """Alias para compatibilidad."""
        return self.stage_config

    def execute(self, manager: DataFormatter) -> Tuple[Optional[DataFormatter], float]:
        """Ejecuta la fase de OCR completa."""
        return self.run_ocr_on_polygons(manager)
        
    def run_ocr_on_polygons(self, manager: DataFormatter) -> Tuple[Optional[DataFormatter], float]:
        start_time = time.time()
        for worker_idx, worker in enumerate(self.workers):
            worker_start = time.time()
            worker_name = worker.__class__.__name__
            logger.debug(f"Ejecutando Worker {worker_idx + 1}/{len(self.workers)}: {worker_name}")
            
            context: Dict[str, Any] = {
                    "worker_name": worker_name,
                    "output_paths": self.output_paths,
                    "project_root": self.project_root
                }    
                
            if not worker.transcribe(context, manager):
                logger.error(f"Fallo en OCR: {worker.__class__.__name__}")
                return None, 0.0

            worker_time = time.time() - worker_start
            logger.debug(f"Worker {worker.__class__.__name__} completado en: {worker_time:.6f}s")
        ocr_time = time.time() - start_time
        return manager, ocr_time
    