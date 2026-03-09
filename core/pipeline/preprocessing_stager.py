# PerfectOCR/core/coordinators/preprocessing_coordinator.py
import logging
import time
from typing import Any, Dict, Tuple, Optional
from core.domain.data_formatter import DataFormatter
from core.factory.abstract_stager import AbstractStager

logger = logging.getLogger(__name__)

class PreprocessingStager(AbstractStager):
    """Coordina la fase de preprocesamiento, delegando todo el trabajo a un único worker autosuficiente."""

    def execute(self, manager: DataFormatter, context: Optional[Dict[str, Any]] = None) -> Tuple[Optional[DataFormatter], float]:
        """Ejecuta la fase de preprocesamiento completa."""
        return self.apply_preprocessing_pipelines(manager, context)

    def apply_preprocessing_pipelines(self, manager: DataFormatter, context: Optional[Dict[str, Any]] = None) -> Tuple[Optional[DataFormatter], float]:
        start_time = time.time()
        
        # Base context setup
        exec_context: Dict[str, Any] = context.copy() if context else {}
        if "output_paths" not in exec_context:
            exec_context["output_paths"] = self.output_paths
        if "project_root" not in exec_context:
            exec_context["project_root"] = self.project_root

            # Para cada worker, procesar todos los polígonos
        for worker_idx, worker in enumerate(self.workers):
            worker_start = time.time()
            worker_name = worker.__class__.__name__
            logger.debug(f"Worker {worker_idx + 1}/{len(self.workers)}: {worker_name}")
                    
            exec_context["worker_name"] = worker_name
                
            # Worker procesa esta imagen específica
            if not worker.preprocess(exec_context, manager):
                logger.error(f"Worker {worker_name} falló", exc_info=True)
                return None, 0.0

            worker_time = time.time() - worker_start
            logger.debug(f"Worker {worker.__class__.__name__} completado en: {worker_time:.6f}s")

        elapsed = time.time() - start_time
        logger.debug(f"Preprocesamiento completado en: {elapsed:.6f}s")
        return manager, elapsed
        