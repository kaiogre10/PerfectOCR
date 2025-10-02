# PerfectOCR/core/pipeline/input_stager.py
import logging
import time
from typing import Optional, Tuple, Dict, Any
from core.factory.abstract_stager import AbstractStager
from core.domain.data_formatter import DataFormatter

logger = logging.getLogger(__name__)

class ImagePreparationStager(AbstractStager):
    """Stager de preparación de imágenes."""
    
    def execute(self, manager: DataFormatter) -> Tuple[Optional[DataFormatter], float]:
        """Ejecuta la fase de preparación completa."""
        start_time = time.time()
        
        context: Dict[str, Any] = {
            "output_paths": self.output_paths,
            "project_root": self.project_root
        }
        
        for worker in self.workers:
            worker_start = time.time()
            if not worker.process(context, manager):
                logger.error(f"[ImagePrepStager] Fallo en {worker.__class__.__name__}")
                return None, 0.0
            
            worker_time = time.time() - worker_start
            logger.debug(f"[ImagePrepStager] {worker.__class__.__name__} completado en {worker_time:.3f}s")
        
        total_time = time.time() - start_time
        logger.debug(f"[ImagePrepStager] Completado en {total_time:.3f}s")
        return manager, total_time