# PerfectOCR/coordinators/tensoring_coordinator.py
import logging
import time
from typing import Any, Dict, Tuple, Optional
from core.domain.data_formatter import DataFormatter
from core.factory.abstract_stager import AbstractStager

logger = logging.getLogger(__name__)

class VectorizationStager(AbstractStager):
    """Inicializa el coordinador y sus workers. """
    
    def execute(self, manager: DataFormatter, context: Optional[Dict[str, Any]] = None) -> Tuple[Optional[DataFormatter], float]:
        """Ejecuta la fase de vectorización completa."""
        return self.vectorize_results(manager, context)
    
    def vectorize_results(self, manager: DataFormatter, context: Optional[Dict[str, Any]] = None) -> Tuple[Optional[DataFormatter], float]:
        """
        Orquesta el flujo completo de vectorización siguiendo una estrategia por fases
        para máxima eficiencia de memoria.
        """ 
        start_time = time.perf_counter()
        try:
            exec_context: Dict[str, Any] = context.copy() if context else {}
            if "output_paths" not in exec_context:
                exec_context["output_paths"] = self.output_paths
            if "project_root" not in exec_context:
                exec_context["project_root"] = self.project_root

            for worker_idx, worker in enumerate(self.workers):
                worker_start = time.perf_counter()
                worker_name = worker.__class__.__name__
                logger.debug(f"Inicia Worker: {worker_idx + 1}/{len(self.workers)}: {worker_name}")

                exec_context["worker_name"] = worker_name  # Actualiza el nombre en cada iteración

                if not worker.vectorize(exec_context, manager):
                    logger.error(f"Worker {worker_name} falló o devolvió resultados vacíos", exc_info=True)
                    return None, 0.0
                if manager.workflow:
                    logger.debug(f"Worker {worker_name} completado en: {time.perf_counter() - worker_start:.6f}s")
                # continue no es necesario aquí

            
            logger.debug(f"Etapa 4 completado en: {time.perf_counter() - start_time:.6f}s")
            return manager, time.perf_counter() - start_time
        except Exception as e:
            logger.error(f"Error en vectorización: '{e}'", exc_info=True)
        return None, time.perf_counter() - start_time