# core/workers/ocr/text_refiner.py
from typing import Dict, Any, Optional, List, Tuple
from domain.data_formatter import DataFormatter
from domain.abstract_worker import OCRAbstractWorker
from core.workers.ocr.text_cleaner import TextCleaner
from core.workers.ocr.fragmenter import Fragmenter
from core.workers.ocr.text_corrector import TextCorrector
from services.output_service import save_text_debug
from utils.text_utils import clasify_words, get_cuants, contains_quantitative, find_key_data, find_umd, contains_umd
import logging
import time

logger = logging.getLogger(__name__)

class Refiner(OCRAbstractWorker):
    """
    Orquesta un ciclo de refinamiento de texto post-OCR con clasificación selectiva optimizada.
    """
    def __init__(self, config: Dict[str, Any], project_root: str, cleaner: Optional[TextCleaner] = None, corrector: Optional[TextCorrector] = None, fragmenter: Optional[Fragmenter] = None):
        super().__init__(config, project_root)
        self.cleaner = cleaner
        self.fragmenter = fragmenter
        self.corrector = corrector
        self.output = config.get("cleanned_text")
        worker_config = config.get("text_refine", {})
        self.seman_clas_log = config.get("seman_clas")
        self.refined_text = config.get("refined_text")
        self.semantic_types_log = list(range(0, 6)) if -1 in config["semantic_types_log"] else config["semantic_types_log"]
        self.num_passes = worker_config.get("num_passes")

    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """Ejecuta el ciclo de refinamiento con clasificación selectiva."""
        try:
            t0 = time.perf_counter()

            self.get_early_data(manager)
            self.preprocess_text(manager)
            self.classify_strings(manager)
            
            for _ in range(self.num_passes):
                if self.cleaner:
                    self.cleaner.transcribe(context, manager)
                    self.classify_strings(manager)

                if self.corrector:
                    self.corrector.transcribe(context, manager)
                    self.classify_strings(manager)
                
                if self.fragmenter:
                    self.fragmenter.transcribe(context, manager)
                    self.classify_strings(manager)

            if self.seman_clas_log or self.output:
                logger.info(f"Tiempo de refinado: {time.perf_counter() - t0:.6f}")
                polygons = manager.workflow.polygons if manager.workflow else {}
                if polygons is None:
                    logger.error(f"NO HAY POLÍGONOS DESPUÉS DE REFINAMINTO TEXTUAL")
                    return False

                poly_output: Dict[str, Any]= {}
                for poly, poly_data in polygons.items():
                    text = poly_data.ocr_text or ""
                    s_clas = poly_data.semantic_clasification
                    poly_output[poly] = {"text": text, "sc": s_clas}
                    if self.seman_clas_log:
                        if any(sc in self.semantic_types_log for sc in s_clas):
                            logger.info(f"{poly}: '{text}', clas: {s_clas} | t_cuant: {poly_data.cuant_chars}")

                if self.output:
                    file_name: str = manager.workflow.metadata.image_name if manager.workflow else ""
                    save_text_debug(poly_output, file_name)

            return True
        except Exception as e:
            logger.error(f"Error refinando texto: {e}", exc_info=True)
            context={}
        return False
        
    def classify_strings(self, manager: DataFormatter) -> bool:
        try:
            if not manager.workflow or not manager.workflow.polygons:
                logger.warning("Semantic Clasificator no tiene polígonos para procesar")
                return False
                
            polygons_to_classify = manager.workflow.polygons if manager.workflow else {}
            
            if not polygons_to_classify:
                logger.warning("No hay polígonos que clasificar")
                return False
            
            # Clasificar solo los polígonos seleccionados
            # t0 = time.perf_counter()
            final_results: Dict[str, Tuple[List[int], int]] = clasify_words(polygons_to_classify)
            # logger.info(f"Tiempo de clasificación: {time.perf_counter() - t0:.6f}'s")
            return manager.update_semantic_clasification(final_results)

        except Exception as e:
            logger.warning(f"Error en el clasificador: {e}", exc_info=True)
            return False
            
    def get_early_data(self, manager: DataFormatter) -> bool:
        """Asigna key_field sobre texto OCR crudo antes de fragmentar/clasificar. Cada tipo de dato se marca como mucho una vez por documento (orden lectura: poly_index)."""
        if not manager.workflow or not manager.workflow.polygons:
            logger.warning("Semantic Clasificator no tiene polígonos para procesar")
            return False

        polygons = manager.workflow.polygons if manager.workflow else {}
        if not polygons:
            logger.error("No hay polygons para procesar", exc_info=True)
            return False
        # [fecha, rfc, iva] — ya satisfechos en el documento
        state: List[bool] = [False, False, False, False, False, False, False, False]

        for _, pd in polygons.items():
            kf = pd.key_field
            if kf is None:
                continue
            if 9 in kf:
                state[0] = True
            if 7 in kf:
                state[1] = True
            if 8 in kf:
                state[2] = True
            if 10 in kf:
                state[3] = True
            if 11 in kf:
                state[4] = True
            if 0 in kf:
                state[5] = True
            if 12 in kf:
                state[6] = True

        polygon_updates: Dict[str, List[int]] = {}

        for poly_id, poly_data in polygons.items():
            if poly_data.key_field is not None:
                continue

            text = poly_data.ocr_text or ""
            key_field = find_key_data(text, state)

            if key_field is None:
                continue
            
            polygon_updates[poly_id] = [key_field]

        if polygon_updates:
            return manager.update_key_field(polygon_updates)
            
        return False
            
    def preprocess_text(self, manager: DataFormatter) -> bool:
        if not manager.workflow or not manager.workflow.polygons:
            logger.warning("Semantic Clasificator no tiene polígonos para procesar")
            return False
            
        polygons = manager.workflow.polygons if manager.workflow else {}
        if not polygons:
            logger.error("No hay polygons para procesar")
            return False
        
        final_polygons: Dict[str, Dict[str, Any]] = {}
        
        for poly, poly_data in polygons.items():
            text = poly_data.ocr_text or ""
            sc = poly_data.semantic_clasification
            kf = poly_data.key_field
            if not text:
                final_polygons[poly] = {"text": text}
                continue

            if 0 in sc and kf is not None:
                final_polygons[poly] = {"text": text}
                continue

            elif len(text) < 2:
                final_polygons[poly] = {"text": text}
                continue
            
            elif text.isalpha():
                final_polygons[poly] = {"text": text}
                continue
                
            if contains_quantitative(text):
                qtext = get_cuants(text)
                # logger.info(f"POTENCIAL CUANTS: '{text}' -> '{qtext}'")
                if qtext != text and self.refined_text:
                    logger.info(f"CUANT ENCONTRADO: '{poly}' | Texy: '{text}' -> '{set(text.split(" ")).difference(set(qtext.split(" ")))}' → '{qtext}'")
                text = qtext

            if contains_umd(text):
                umd_text = find_umd(text)
                # logger.info(f"POTENCIAL UMDS: {umd_text != text} -> '{text}'")
                if umd_text != text:
                    if self.refined_text:
                        logger.info(f"UMD ENCONTRADA:'{poly}' | Text: '{text}' -> '{set(text.split(" ")).difference(set(umd_text.split(" ")))}' → '{umd_text}'")
                    final_polygons[poly] = {"text": umd_text}
                    continue

                text = umd_text
                final_polygons[poly] = {"text": text}
                continue
            else:
                final_polygons[poly] = {"text": text}
                continue

        return manager.update_ocr_results(final_polygons, "refiner")