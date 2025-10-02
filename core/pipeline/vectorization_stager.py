# PerfectOCR/coordinators/tensoring_coordinator.py
import logging
import time
import pandas as pd # type: ignore
from typing import Any, Dict, Tuple, Optional
from core.domain.data_formatter import DataFormatter
from core.factory.abstract_stager import AbstractStager
from core.domain.data_models import Polygons, Metadata

logger = logging.getLogger(__name__)

class VectorizationStager(AbstractStager):
    """Inicializa el coordinador y sus workers. """

    @property
    def config(self):
        """Alias para compatibilidad."""
        return self.stage_config

    def execute(self, manager: DataFormatter) -> Tuple[Optional[DataFormatter], float]:
        """Ejecuta la fase de vectorización completa."""
        return self.vectorize_results(manager)
    
    def vectorize_results(self, manager: DataFormatter) -> Tuple[Optional[DataFormatter], float]:
        """
        Orquesta el flujo completo de vectorización siguiendo una estrategia por fases
        para máxima eficiencia de memoria.
        """        
        start_time = time.time()
        logger.debug("[VectorStager] Iniciando pipeline de vectorización")
        metadata: Metadata = manager.workflow.metadata
        if not metadata or not manager.workflow:
            logger.warning("No metadata")
        polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
                
        # Para cada worker, procesar todos los polígonos
                                
        context: Dict[str, Any] = {
            "config": self.config,
            "polygons": polygons,
            "metadata": metadata,
            "output_paths": self.output_paths,
            "project_root": self.project_root,
        }
                
        for worker_idx, worker in enumerate(self.workers):
            worker_name = worker.__class__.__name__
            logger.debug(f"[VectorStager] Worker {worker_idx + 1}/{len(self.workers)}: {worker_name}")
            
            result = worker.vectorize(context, manager)
            if result is None or (isinstance(result, pd.DataFrame) and result.empty):
                logger.error(f"Worker {worker_name} falló o devolvió resultados vacíos")
                return None, 0.0
            else:
                # El worker puede haber modificado el contexto, mantenerlo para el siguiente
                # Actualizar polígonos del manager si han sido modificados
                if manager.workflow:
                    context["polygons"] = manager.workflow.polygons
                continue
    
        vect_time = time.time() - start_time
        logger.debug(f"[VectorStager] Pipeline completado en: {vect_time:.6f}s")
        return manager, vect_time