#core/workers/ocr/text_refiner.py
import logging
from typing import Dict, Any
from core.domain.data_formatter import DataFormatter
from core.factory.abstract_worker import OCRAbstractWorker
from core.workers.ocr.semantic_clasificator import SemanticClasificator
from core.workers.ocr.text_cleaner import TextCleaner
from core.workers.ocr.fragmenter import Fragmenter

logger = logging.getLogger(__name__)

class Refiner(OCRAbstractWorker):
    """
    Orquesta un ciclo de refinamiento de texto post-OCR.
    Ejecuta clasificación, limpieza y fragmentación en un número fijo de pasadas
    para estabilizar el texto antes de la búsqueda de datos.
    """
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.worker_config = self.config
        self.num_passes = self.worker_config.get("num_passes", 2)
        self.clasificator = SemanticClasificator(config, project_root)
        self.cleaner = TextCleaner(config, project_root)
        self.fragmenter = Fragmenter(config, project_root)
        logger.info(f"TextRefinerWorker inicializado para {self.num_passes} pasadas.")

    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """
        Ejecuta el ciclo de refinamiento.
        """
        try:
            for i in range(self.num_passes):
                pass_num = i + 1
                logger.info(f"--- Iniciando Pasada de Refinamiento de Texto #{pass_num} ---")

                # 1. Clasificar todo el texto actual
                logger.info(f"Pasada #{pass_num}: Clasificación Semántica")
                self.clasificator.transcribe(context, manager)

                # 2. Limpiar polígonos basura
                logger.info(f"Pasada #{pass_num}: Limpieza de Texto")
                self.cleaner.transcribe(context, manager)

                # 3. Fragmentar polígonos que lo necesiten
                logger.info(f"Pasada #{pass_num}: Fragmentación de Texto")
                self.fragmenter.transcribe(context, manager)
                
                logger.info(f"--- Finalizada Pasada de Refinamiento #{pass_num} ---")

            # Clasificación final post-fragmentación
            logger.info("--- Iniciando Clasificación Semántica Final ---")
            self.clasificator.transcribe(context, manager)
            logger.info("--- Clasificación Semántica Final Completada ---")

            return True
        
        except Exception as e:
            logger.error(f"Error durante el refinamiento de texto: {e}", exc_info=True)
            return False