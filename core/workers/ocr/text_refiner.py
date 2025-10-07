#core/workers/ocr/text_refiner.py
import logging
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
            for i in range(num_passes):
                pass_num = i + 1
                logger.debug(f"--- Iniciando Pasada de Refinamiento de Texto #{pass_num} ---")

                # 1. Clasificar todo el texto actual
                logger.debug(f"Pasada #{pass_num}: Clasificación Semántica")
                self.clasificator.transcribe(context, manager)

                # 2. Limpiar polígonos basura
                logger.debug(f"Pasada #{pass_num}: Limpieza de Texto")
                self.cleaner.transcribe(context, manager)

                # 3. Fragmentar polígonos que lo necesiten
                logger.debug(f"Pasada #{pass_num}: Fragmentación de Texto")
                self.fragmenter.transcribe(context, manager)
                                
                logger.debug(f"--- Finalizada Pasada de Refinamiento #{pass_num} ---")

            # 4. Correciones puntuales
            logger.debug(f" INICIANDP Corrección de Texto")
            self.corrector.transcribe(context, manager)
            
            # Clasificación final post-fragmentación
            logger.debug("--- Iniciando Clasificación Semántica Final ---")
            self.clasificator.transcribe(context, manager)
            logger.debug("--- Clasificación Semántica Final Completada ---")

            return True
        
        except Exception as e:
            logger.error(f"Error durante el refinamiento de texto: {e}", exc_info=True)
            return False