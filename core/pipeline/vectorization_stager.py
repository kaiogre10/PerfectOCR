# PerfectOCR/coordinators/tensoring_coordinator.py
import logging
import time
from typing import Any, Dict, Tuple, Optional
from core.domain.data_formatter import DataFormatter
from core.factory.abstract_stager import AbstractStager

logger = logging.getLogger(__name__)

class VectorizationStager(AbstractStager):
    """Inicializa el coordinador y sus workers. """
    
    def execute(self, manager: DataFormatter) -> Tuple[Optional[DataFormatter], float]:
        """Ejecuta la fase de vectorización completa."""
        return self.vectorize_results(manager)
    
    def vectorize_results(self, manager: DataFormatter) -> Tuple[Optional[DataFormatter], float]:
        """
        Orquesta el flujo completo de vectorización siguiendo una estrategia por fases
        para máxima eficiencia de memoria.
        """ 
        start_time = time.time()       
        context: Dict[str, Any] = {
            "output_paths": self.output_paths,
            "project_root": self.project_root,
        }

        for worker_idx, worker in enumerate(self.workers):
            worker_start = time.time()
            worker_name = worker.__class__.__name__
            logger.debug(f"Inicia Worker: {worker_idx + 1}/{len(self.workers)}: {worker_name}")

            context["worker_name"] = worker_name  # Actualiza el nombre en cada iteración

            if not worker.vectorize(context, manager):
                logger.error(f"Worker {worker_name} falló o devolvió resultados vacíos")
                return None, 0.0
            if manager.workflow:
                worker_time = time.time() - worker_start
                logger.debug(f"Worker {worker_name} completado en: {worker_time:.6f}s")
            # continue no es necesario aquí

        vect_time = time.time() - start_time
        logger.debug(f"Etapa 4 completado en: {vect_time:.6f}s")
        return manager, vect_time