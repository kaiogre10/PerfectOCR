# PerfectOCR/coordinators/tensoring_coordinator.py
import logging
import time
import pandas as pd
from typing import Any, Dict, Tuple, List, Optional
from core.domain.data_formatter import DataFormatter
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.domain.data_models import Polygons, Metadata

logger = logging.getLogger(__name__)

class VectorizationStager:
    """Inicializa el coordinador y sus workers. """
    def __init__(self, workers: List[VectorizationAbstractWorker], stage_config: Dict[str, Any], output_paths: Optional[List[str]], project_root: str):
        self.project_root = project_root
        self.workers = workers
        self.stage_config = stage_config
        self.output_paths = output_paths
    
    def vectorize_results(self, manager: DataFormatter) -> Tuple[Optional[DataFormatter], float]:
        """
        Orquesta el flujo completo de vectorización siguiendo una estrategia por fases
        para máxima eficiencia de memoria.
        """        
        start_time = time.time()
        logger.debug("[VectorStager] Iniciando pipeline de vectorización")
        metadata: Dict[str, Metadata] = manager.workflow.metadata if manager.workflow else {}
        polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
        image_name: str = metadata.image_name if manager.workflow else {}
                
        # Para cada worker, procesar todos los polígonos
        for worker_idx, worker in enumerate(self.workers):
            worker_name = worker.__class__.__name__
            logger.debug(f"[VectorStager] Worker {worker_idx + 1}/{len(self.workers)}: {worker_name}")
                                
            context: Dict[str, Any] = {
                "polygons": polygons,
                "metadata": metadata,
                "output_paths": self.output_paths,
                "project_root": self.project_root,
                "image_name": image_name
            }
                
            result = worker.vectorize(context, manager)
            if result is None or (isinstance(result, pd.DataFrame) and result.empty):
                # El resultado es None o un DataFrame vacío
                # Maneja el caso de error o datos insuficientes
                logger.error(f"Worker {worker_name} falló o devolvió resultados vacíos")
                return None, 0.0
            else:
                continue
        
        vect_time = time.time() - start_time
        logger.info(f"[VectorStager] Pipeline completado en: {vect_time:.6f}s")
        return manager, vect_time