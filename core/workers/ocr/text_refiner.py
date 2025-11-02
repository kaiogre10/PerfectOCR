#core/workers/ocr/text_refiner.py
import logging
import time
from typing import Dict, Any
from core.domain.data_formatter import DataFormatter
from core.factory.abstract_worker import OCRAbstractWorker
from core.workers.ocr.semantic_clasificator import SemanticClasificator
from core.workers.ocr.text_cleaner import TextCleaner
from core.workers.ocr.text_corrector import TextCorrector
from core.workers.ocr.fragmenter import Fragmenter

logger = logging.getLogger(__name__)

class Refiner(OCRAbstractWorker):
    """
    Orquesta un ciclo de refinamiento de texto post-OCR con clasificación selectiva optimizada.
    """
    def __init__(self, config: Dict[str, Any], project_root: str, clasificator: SemanticClasificator, cleaner: TextCleaner, fragmenter: Fragmenter, corrector: TextCorrector):
        super().__init__(config, project_root)
        self.worker_config = self.config.get("text_refiner", {})
        self.num_passes = self.worker_config.get("num_passes")
        self.clasificator = clasificator
        self.cleaner = cleaner
        self.fragmenter = fragmenter
        self.corrector = corrector

    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """
        Ejecuta el ciclo de refinamiento con clasificación selectiva.
        """
        start_time = time.perf_counter()
        logger.info(f"Refinador inicializado para {self.num_passes} pasadas.")

        try:
            for i in range(self.num_passes):
                pass_num = i + 1
                logger.info(f"Iniciando Bucle de Refinamiento de Texto #{pass_num}")
                
                # Determinar si usar filtro selectivo (solo en pasadas 2+)
                use_filter = (i > 0)
        
                logger.info(f"Pasada 1, bucle #{pass_num}: Clasificación Semántica (filtro={use_filter})")
                self.clasificator.transcribe(context, manager, filter_modified=use_filter)
                
                logger.info(f"Bucle #{pass_num}: Limpieza de Texto")
                self.cleaner.transcribe(context, manager)

                logger.info(f"Pasada 2, bucle #{pass_num}: Clasificación Semántica (solo corregidos)")
                self.clasificator.transcribe(context, manager, filter_modified=True)
                
                logger.info(f"Bucle #{pass_num}: Fragmentación de Texto")
                self.fragmenter.transcribe(context, manager)

                logger.info(f"Pasada 3, bucle #{pass_num}: Clasificación Semántica (solo limpiados)")
                self.clasificator.transcribe(context, manager, filter_modified=True)
                
                logger.info(f"Bucle #{pass_num}: Corrección textual")
                self.corrector.transcribe(context, manager)

            # Clasificación final completa para asegurar consistencia
            logger.info(f"Pasada final: Clasificación Semántica (completa)")
            self.clasificator.transcribe(context, manager, filter_modified=False)
            
            logger.info(f"Limpieza textual completa en: {time.perf_counter()-start_time:.6f}s")
            return True
        
        except Exception as e:
            logger.error(f"Error durante el refinamiento de texto: {e}", exc_info=True)
            return False
