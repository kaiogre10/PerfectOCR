# PerfectOCR/core/coordinators/ocr_manager.py
import time
import logging
from typing import Optional, Dict, Any, Tuple, List
from core.domain.data_formatter import DataFormatter
from core.factory.abstract_worker import OCRAbstractWorker

logger = logging.getLogger(__name__)

class OCRStager:
    def __init__(self, workers: List[OCRAbstractWorker], stage_config: Dict[str, Any], output_paths: Optional[List[str]], project_root: str):
        self.workers = workers
        self.config = stage_config
        self.project_root = project_root
        self.output_paths = output_paths if output_paths is not None else []
        # self._text_cleanner = TextCleaner(config=self.config, project_root=self.project_root)
        # self._interceptor = Interceptor(config=self.config, project_root=self.project_root)
        
    def run_ocr_on_polygons(self, manager: DataFormatter) -> Tuple[Optional[DataFormatter], float]:
        start_time = time.perf_counter()
        
        # Verificar que tenemos workers de OCR
        if not self.workers:
            logger.error("OCRStager: No hay workers de OCR disponibles")
            return None, 0.0
        
        ocr_worker = self.workers[0]  # Asumiendo que solo hay uno
        if not hasattr(ocr_worker, 'transcribe'):
            logger.error("OCRStager: El worker no implementa el método transcribe")
            return None, 0.0
            
        try:
            context = {
                "config": self.config
            }
            
            # Ejecutar transcripción
            success = ocr_worker.transcribe(context, manager)
            if not success:
                logger.error("OCRStager: Fallo en la transcripción OCR")
                return None, 0.0
                        
            ocr_time = time.perf_counter() - start_time
            logger.info(f"OCR completado en {ocr_time:.4f}s")
            return manager, ocr_time
            
        except Exception as e:
            logger.error(f"Error en OCRStager: {e}", exc_info=True)
            return None, 0.0