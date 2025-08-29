# PerfectOCR/core/coordinators/preprocessing_coordinator.py
import logging
import time
from typing import Any, Dict, Tuple, List, Optional
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons, Metadata
from core.factory.abstract_worker import PreprocessingAbstractWorker

logger = logging.getLogger(__name__)

class PreprocessingStager:
    """
    Coordina la fase de preprocesamiento, delegando todo el trabajo a un único worker autosuficiente.
    """
    def __init__(self, workers: List[PreprocessingAbstractWorker], stage_config: Dict[str, Any], output_paths: Optional[List[str]], project_root: str):
        self.project_root = project_root
        self.workers = workers
        self.stage_config = stage_config
        self.output_paths = output_paths

    def apply_preprocessing_pipelines(self, manager: DataFormatter) -> Tuple[Optional[DataFormatter], float]:
        start_time = time.time()
        logger.debug("[PreprocessingManager] Iniciando pipeline secuencial directo")
        
        # Obtener datos
        metadata: Dict[str, Metadata] = manager.workflow.metadata if manager.workflow else {}
        polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
        
        # Para cada worker, procesar todos los polígonos
        for worker_idx, worker in enumerate(self.workers):
            worker_start = time.time()
            worker_name = worker.__class__.__name__
            logger.debug(f"[PreprocessingStager] Worker {worker_idx + 1}/{len(self.workers)}: {worker_name}")
                    
                # Contexto individual para cada polígono
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
            logger.debug(f"[PreprocessingStager] Worker {worker.__class__.__name__} completado en: {worker_time:.3f}s")

            # Sincronizar resultados de preprocesamiento al manager
            for poly_id, polygon in context["polygons"].items():
                # Obtener posible imagen modificada o None
                cropped_img = polygon.cropped_img.cropped_img if polygon.cropped_img else None
                manager.update_preprocessing_result(poly_id, cropped_img, worker_name, True)
            # Refrescar polígonos desde el manager para la siguiente etapa
            polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
            context["polygons"] = polygons

        
        elapsed = time.time() - start_time
        logger.info(f"[PreprocessingStager] Pipeline completado en: {elapsed:.3f}s; polígonos: {len(polygons)}")
        return manager, elapsed