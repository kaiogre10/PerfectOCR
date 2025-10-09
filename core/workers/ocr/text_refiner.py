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
    Orquesta un ciclo de refinamiento de texto post-OCR.
    """
    def __init__(self, config: Dict[str, Any], project_root: str, 
                 clasificator: SemanticClasificator, 
                 cleaner: TextCleaner, 
                 fragmenter: Fragmenter,
                 corrector: TextCorrector):
        super().__init__(config, project_root)
        self.worker_config = self.config.get("text_refiner", {})
        self.clasificator = clasificator
        self.cleaner = cleaner
        self.fragmenter = fragmenter
        self.corrector = corrector

    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """
        Ejecuta el ciclo de refinamiento.
        """
        num_passes = self.worker_config.get("num_passes", {})
        logger.debug(f"Refinador inicializado para {num_passes} pasadas.")
        try:
            t0 = time.perf_counter()
            for i in range(num_passes):
                pass_num = i + 1
                logger.debug(f"Iniciando Bucle de Refinamiento de Texto #{pass_num}")

                # 3. Clasificar todo el texto limpio
                logger.debug(f"Pasada 2, bucle: #{pass_num}: Clasificación Semántica")
                self.clasificator.transcribe(context, manager)

                logger.debug(f"Bucle: #{pass_num}: Correción textual")
                self.corrector.transcribe(context, manager)
                
                # 1. Clasificar todo el texto actual
                logger.debug(f"Pasada 1, bucle: #{pass_num}: Clasificación Semántica")
                self.clasificator.transcribe(context, manager)

                # 2. Limpiar polígonos basura
                logger.debug(f"Bucle: #{pass_num}: Limpieza de Texto")
                self.cleaner.transcribe(context, manager)


                logger.debug(f"Pasada 3, bucle: #{pass_num}: Clasificación Semántica")
                self.clasificator.transcribe(context, manager)

                # 4. Fragmentar polígonos que lo necesiten
                logger.debug(f"Bucle: #{pass_num}: Fragmentación de Texto")
                self.fragmenter.transcribe(context, manager)
    
            logger.debug(f"Pasada final Clasificación Semántica")

            self.clasificator.transcribe(context, manager)                
            logger.debug(f"Clasificación Semántica Final Completada en: {time.perf_counter()-t0:.6f}s")

            return True
        
        except Exception as e:
            logger.error(f"Error durante el refinamiento de texto: {e}", exc_info=True)
            return False