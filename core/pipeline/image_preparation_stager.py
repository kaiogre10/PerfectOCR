#PerfectOCR/core/pipeline/input_stager.py
import logging
import time
from typing import Optional, Dict, Any
from core.contracts.abstract_stager import AbstractStager
from domain.data_formatter import DataFormatter

logger = logging.getLogger(__name__)

class ImagePreparationStager(AbstractStager):
    """Stager de preparación de imágenes."""
    def execute(self, manager: DataFormatter, context: Optional[Dict[str, Any]] = None) -> Optional[DataFormatter]:
        # Usar contexto base si existe, sino crear uno nuevo
        try:
            for worker_idx, worker in enumerate(self.workers):
                worker_name = worker.__class__.__name__
                logger.debug(f"Ejecutando {worker_idx + 1}/{len(self.workers)}: '{worker_name}'")
                worker_start = time.perf_counter()
                if not worker.process(context, manager):
                    logger.error(f"'{worker_name}' falló, tiempo: {time.perf_counter() - worker_start:.6f}'s", exc_info=True)
                    return None

                if self.modules_config.get("time_worker_log"):
                    logger.info(f"'{worker_name}' completado en: {time.perf_counter() - worker_start:.6f}'s")

            return manager
        except Exception as e:
            logger.error(f"ERROR EN FASE INICIAL: {e}", exc_info=True)
        return None