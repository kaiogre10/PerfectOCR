# PerfectOCR/core/coordinators/preprocessing_coordinator.py
import logging
import time
from typing import Any, Dict, Tuple, Optional
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons, Metadata
from core.factory.abstract_stager import AbstractStager

logger = logging.getLogger(__name__)

class PreprocessingStager(AbstractStager):
    """
    Coordina la fase de preprocesamiento, delegando todo el trabajo a un único worker autosuficiente.
    """

    def execute(self, manager: DataFormatter) -> Tuple[Optional[DataFormatter], float]:
        """Ejecuta la fase de preprocesamiento completa."""
        return self.apply_preprocessing_pipelines(manager)

    def apply_preprocessing_pipelines(self, manager: DataFormatter) -> Tuple[Optional[DataFormatter], float]:
        start_time = time.time()
        logger.debug("[PreprocessingManager] Iniciando pipeline secuencial directo")
        
        # Obtener datos
        metadata: Metadata = manager.workflow.metadata if manager.workflow else {}
        polygons = manager.workflow.polygons if manager.workflow else {}
        
        # Para cada worker, procesar todos los polígonos
        for worker_idx, worker in enumerate(self.workers):
            worker_start = time.time()
            worker_name = worker.__class__.__name__
            logger.debug(f"[PreprocessingStager] Worker {worker_idx + 1}/{len(self.workers)}: {worker_name}")
                    
            context: Dict[str, Any] = {
                "polygons": polygons,
                "metadata": metadata,
                "output_paths": self.output_paths,
                "project_root": self.project_root
            }
                
            # Worker procesa esta imagen específica
            if not worker.preprocess(context, manager):
                logger.error(f"Worker {worker_name} falló")
                return None, 0.0

            worker_time = time.time() - worker_start
            logger.debug(f"[PreprocessingStager] Worker {worker.__class__.__name__} completado en: {worker_time:.6f}s")

            # Sincronizar resultados de preprocesamiento al manager
            for poly_id, polygon in context["polygons"].items():
                # Obtener posible imagen modificada o None
                cropped_img = polygon.cropped_img.cropped_img if polygon.cropped_img else None
                manager.update_preprocessing_result(poly_id, cropped_img, worker_name)

            polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
            context["polygons"] = polygons

        elapsed = time.time() - start_time
        logger.debug(f"[PreprocessingStager] Preprocesamiento completado en: {elapsed:.6f}s; polígonos: {len(polygons)}")
        return manager, elapsed