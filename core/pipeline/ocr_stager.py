# PerfectOCR/core/coordinators/ocr_manager.py
import time
import logging
from typing import Optional, Dict, Any, Tuple, List
from core.domain.data_formatter import DataFormatter
from core.factory.abstract_worker import OCRAbstractWorker

logger = logging.getLogger(__name__)

class OCRStager:
    def __init__(self, workers: List[OCRAbstractWorker], stage_config: Dict[str, Any], output_paths: Optional[List[str]], project_root: str):
        self.project_root = project_root
        self.workers = workers
        self.config = stage_config
        self.output_paths = output_paths if output_paths is not None else []
        
    def run_ocr_on_polygons(self, manager: DataFormatter) -> Tuple[Optional[DataFormatter], float]:
        start_time = time.time()
                
        for worker_idx, worker in enumerate(self.workers):
            worker_start = time.time()
            worker_name = worker.__class__.__name__
            logger.debug(f"OCRStager: Ejecutando worker OCR Worker {worker_idx + 1}/{len(self.workers)}: {worker_name}")
            
            context: Dict[str, Any] = {
                    # "poly_id": poly_id,
                    "config": self.config,
                    "output_paths": self.output_paths,
                    "project_root": self.project_root
                }    
                
            if not worker.transcribe(context, manager):
                logger.error(f"OCRStager: Fallo en la transcripción OCR en worker #{worker_idx}: {worker.__class__.__name__}")
                return None, 0.0

            worker_time = time.time() - worker_start
            logger.info(f"[OCRStager] Worker {worker.__class__.__name__} completado en: {worker_time:.6f}s")
        ocr_time = time.time() - start_time
        logger.info(f"OCR completado en {ocr_time:.6f}s")
        return manager, ocr_time
