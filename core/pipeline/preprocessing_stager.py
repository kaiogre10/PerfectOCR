# PerfectOCR/core/coordinators/preprocessing_coordinator.py
import logging
import time
from typing import Any, Dict, Tuple, List, Optional
from core.domain.data_formatter import DataFormatter
from core.factory.abstract_worker import PreprossesingAbstractWorker

logger = logging.getLogger(__name__)

class PreprocessingStager:
    """
    Coordina la fase de preprocesamiento, delegando todo el trabajo a un único worker autosuficiente.
    """
    def __init__(self, workers: List[PreprossesingAbstractWorker], stage_config: Dict[str, Any], output_paths: Optional[List[str]], project_root: str):
        self.project_root = project_root
        self.workers = workers
        self.stage_config = stage_config
        self.output_paths = output_paths

    def apply_preprocessing_pipelines(self, manager: DataFormatter) -> Tuple[Optional[DataFormatter], float]:
        start_time = time.time()
        logger.info("[PreprocessingManager] Iniciando pipeline secuencial directo")
        
        # Obtener datos
        metadata = manager.get_metadata()
        polygons = manager.get_polygons_with_cropped_img()
        output_paths = self.output_paths
        
        # Para cada worker, procesar todos los polígonos
        for worker_idx, worker in enumerate(self.workers):
            worker_name = worker.__class__.__name__
            logger.info(f"[PreprocessingManager] Worker {worker_idx + 1}/{len(self.workers)}: {worker_name}")
            
            # Procesar cada polígono con este worker
            for poly_id, poly_data in polygons.items():
                cropped_img = poly_data.get("cropped_img")
                if cropped_img is None:
                    logger.warning(f"Imagen recortada no encontrada para {poly_id}")
                    continue
                    
                # Contexto individual para cada polígono
                context = {
                    "poly_id": poly_id,
                    "cropped_img": cropped_img,
                    "metadata": metadata,
                    "output_paths": self.output_paths,
                    "project_root": self.project_root
                }
                
                # Worker procesa esta imagen específica
                if not worker.preprocess(context, manager):
                    logger.error(f"Worker {worker_name} falló en {poly_id}")
                    return None, 0.0
        
        elapsed = time.time() - start_time
        logger.info(f"[PreprocessingStager] Pipeline completado en: {elapsed:.3f}s; polígonos: {len(polygons)}")
        return manager, elapsed