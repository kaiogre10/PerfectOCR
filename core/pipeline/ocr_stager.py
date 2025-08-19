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
        start_time = time.perf_counter()

        if not self.workers:
            logger.error("OCRStager: No hay workers de OCR disponibles")
            return None, 0.0

        try:
            context = {
                "config": self.config
            }

            for worker_idx, worker in enumerate(self.workers):
                if not hasattr(worker, 'transcribe'):
                    logger.error(f"OCRStager: El worker #{worker_idx} no implementa el método transcribe")
                    return None, 0.0

                logger.info(f"OCRStager: Ejecutando worker OCR #{worker_idx}: {worker.__class__.__name__}")
                success = worker.transcribe(context, manager)
                if not success:
                    logger.error(f"OCRStager: Fallo en la transcripción OCR en worker #{worker_idx}: {worker.__class__.__name__}")
                    return None, 0.0

            ocr_time = time.perf_counter() - start_time
            logger.info(f"OCR completado en {ocr_time:.4f}s")
            return manager, ocr_time

        except Exception as e:
            logger.error(f"Error en OCRStager: {e}", exc_info=True)
            return None, 0.0