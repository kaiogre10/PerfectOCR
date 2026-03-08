# core/workers/ocr/text_refiner.py
from typing import Dict, Any, Optional
from core.domain.data_formatter import DataFormatter
from core.factory.abstract_worker import OCRAbstractWorker
from core.workers.ocr.semantic_clasificator import SemanticClasificator
from core.workers.ocr.text_cleaner import TextCleaner
from core.workers.ocr.text_corrector import TextCorrector
from core.workers.ocr.fragmenter import Fragmenter
import logging

logger = logging.getLogger(__name__)

class Refiner(OCRAbstractWorker):
    """
    Orquesta un ciclo de refinamiento de texto post-OCR con clasificación selectiva optimizada.
    """
    def __init__(self, config: Dict[str, Any], project_root: str, clasificator: SemanticClasificator, cleaner: Optional[TextCleaner] = None, corrector: Optional[TextCorrector] = None, fragmenter: Optional[Fragmenter] = None):
        super().__init__(config, project_root)
        worker_config = config.get("text_refiner", {})
        self.percentile = config["percentile"]
        worker_config["percentile"] = self.percentile 
        self.num_passes = worker_config.get("num_passes")
        self.create_refiners: bool = config.get("create_refiners", False)
        self.output = config.get("cleanned_text")
        self.clasificator = clasificator
        self.cleaner = cleaner
        self.fragmenter = fragmenter
        self.corrector = corrector

    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """
        Ejecuta el ciclo de refinamiento con clasificación selectiva.
        """
        logger.debug(f"Refinador inicializado para {self.num_passes} pasadas.")

        try:
            if 0 >= self.num_passes:
                self.clasificator.transcribe(context, manager, True)
            
            else:
                for i in range(self.num_passes):
                    pass_num = i + 1
                    logger.debug(f"Iniciando Bucle de Refinamiento de Texto #{pass_num}")

                    logger.debug(f"Pasada 1, bucle #{pass_num}: Clasificación Semántica")
                    self.clasificator.transcribe(context, manager, False)

                    if self.fragmenter:
                        logger.debug(f"Bucle #{pass_num}: Fragmentación Textual")
                        self.fragmenter.transcribe(context, manager)

                    logger.debug(f"Pasada 2, bucle #{pass_num}: Clasificación Semántica (solo fragmentados)")
                    self.clasificator.transcribe(context, manager, False)

                    if self.cleaner:
                        logger.debug(f"Bucle #{pass_num}: Limpieza de Texto")
                        self.cleaner.transcribe(context, manager)
                    
                    logger.debug(f"Pasada 3, bucle #{pass_num}: Clasificación Semántica (solo limpiados)")
                    self.clasificator.transcribe(context, manager, False)
                    
                    if self.corrector:
                        logger.debug(f"Bucle #{pass_num}: Corrección textual")
                        self.corrector.transcribe(context, manager)

                logger.debug(f"Pasada final: Clasificación Semántica completa")
                self.clasificator.transcribe(context, manager, True)

            polygons = manager.workflow.polygons if manager.workflow else {}
            for poly, poly_data in polygons.items():
                logger.debug(f"{poly}: '{poly_data.ocr_text}'")

            if self.output:
                from services.output_service import save_raw_json
                file_name: str = manager.workflow.metadata.image_name  # type: ignore
                name = "cleanned_text"
                worker_name = f"{name}" or "refiner"
                output_paths = context["output_paths"]
                polygons = manager.workflow.polygons if manager.workflow else {}
                results: Dict[str, Any] = {}
                for poly_id, polygon in polygons.items():
                    text = getattr(polygon, "ocr_text", None)
                    results[poly_id] = {
                        "text": text,
                    }
                save_raw_json( output_paths, worker_name, results, file_name)

            return True 

        except Exception as e:
            logger.error(f"Error durante el refinamiento de texto: {e}", exc_info=True)
            return False
