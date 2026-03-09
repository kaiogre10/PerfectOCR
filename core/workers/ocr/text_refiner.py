# core/workers/ocr/text_refiner.py
from typing import Dict, Any, Optional, List
from core.domain.data_formatter import DataFormatter
from core.factory.abstract_worker import OCRAbstractWorker
from core.workers.ocr.text_cleaner import TextCleaner
from core.workers.ocr.text_corrector import TextCorrector
from core.workers.ocr.fragmenter import Fragmenter
from core.utils.general_utils import clasify_words
from core.domain.data_models import Polygons
import logging
import time

logger = logging.getLogger(__name__)

class Refiner(OCRAbstractWorker):
    """
    Orquesta un ciclo de refinamiento de texto post-OCR con clasificación selectiva optimizada.
    """
    def __init__(self, config: Dict[str, Any], project_root: str, cleaner: Optional[TextCleaner] = None, corrector: Optional[TextCorrector] = None, fragmenter: Optional[Fragmenter] = None):
        super().__init__(config, project_root)
        self.worker_config = config.get("text_refiner", {})
        self.num_passes = self.worker_config.get("num_passes")
        self.output = config.get("cleanned_text")
        self.cleaner = cleaner
        self.fragmenter = fragmenter
        self.corrector = corrector

    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """
        Ejecuta el ciclo de refinamiento con clasificación selectiva.
        """
        t0 = time.perf_counter()
        try:
            if 0 >= self.num_passes:
                self.classify_strings(manager)
            
            else:
                for i in range(self.num_passes):
                    pass_num = i + 1
                    logger.debug(f"Iniciando Bucle de Refinamiento de Texto #{pass_num}")

                    logger.debug(f"Pasada 1, bucle #{pass_num}: Clasificación Semántica")
                    self.classify_strings(manager)

                    if self.fragmenter:
                        logger.debug(f"Bucle #{pass_num}: Fragmentación Textual")
                        self.fragmenter.transcribe(context, manager)

                    logger.debug(f"Pasada 2, bucle #{pass_num}: Clasificación Semántica (solo fragmentados)")
                    self.classify_strings(manager)

                    if self.cleaner:
                        logger.debug(f"Bucle #{pass_num}: Limpieza de Texto")
                        self.cleaner.transcribe(context, manager)
                    
                    logger.debug(f"Pasada 3, bucle #{pass_num}: Clasificación Semántica (solo limpiados)")
                    self.classify_strings(manager)
                    
                    if self.corrector:
                        logger.debug(f"Bucle #{pass_num}: Corrección textual")
                        self.corrector.transcribe(context, manager)

                logger.debug(f"Pasada final: Clasificación Semántica completa")
                self.classify_strings(manager)

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

            # logger.info(f"Tiempo de refinador: {time.perf_counter() - t0:.6f}'s")
            return True 

        except Exception as e:
            logger.error(f"Error durante el refinamiento de texto: {e}", exc_info=True)
            return False
        
    def classify_strings(self, manager: DataFormatter) -> bool:
        """Clasifica polígonos semánticamente"""
        try:
            if not manager.workflow or not manager.workflow.polygons:
                logger.warning("Semantic Clasificator no tiene polígonos para procesar")
                return False
                
            polygons_to_classify: Dict[str, Polygons] = manager.workflow.polygons
            
            if not polygons_to_classify:
                logger.warning("No hay polígonos que clasificar")
                return True

            # Clasificar solo los polígonos seleccionados
            # t0 = time.perf_counter()
            final_results: Dict[str, int | List[int]] = clasify_words(polygons_to_classify, self.worker_config)
            # logger.info(f"Tiempo de clasificación: {time.perf_counter() - t0:.6f}'s")

            # Actualizar semantic_type Y resetear was_refined si es modo filtrado
            manager.update_semantic_clasification(final_results)

            return True

        except Exception as e:
            logger.warning(f"Error en el clasificador: {e}", exc_info=True)
            return False
