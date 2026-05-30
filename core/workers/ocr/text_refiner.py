# core/workers/ocr/text_refiner.py
from typing import Dict, Any, Optional, List, Tuple
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons
from core.factory.abstract_worker import OCRAbstractWorker
from core.workers.ocr.text_cleaner import TextCleaner
from core.workers.ocr.fragmenter import Fragmenter
from core.workers.ocr.text_corrector import TextCorrector
from core.utils.text_utils import clasify_words, get_cuants, contains_quantitative, find_key_data, find_umd, contains_umd, is_quantitative
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
        self.output = config.get("cleanned_text")
        self.seman_clas_log = config.get("seman_clas")
        semantic_types_log = any(t == -1 for t in config["semantic_types_log"])
        self.semantic_types_log = config["semantic_types_log"] if not semantic_types_log else list(range(6))
        self.num_passes = self.worker_config.get("num_passes")
        self.cleaner = cleaner
        self.fragmenter = fragmenter
        self.corrector = corrector

    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """Ejecuta el ciclo de refinamiento con clasificación selectiva."""
        t0 = time.perf_counter()
        self.preprocess_text(manager)
        self.get_early_data(manager)
        
        if self.num_passes == 0:
            self.classify_strings(manager)
        else:
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

        if self.seman_clas_log:
            logger.info(f"Tiempo de refinado: {time.perf_counter() - t0:.6f}'s")
            polygons = manager.workflow.polygons if manager.workflow else {}
            for poly, poly_data in polygons.items():
                if any(sc in self.semantic_types_log for sc in  poly_data.semantic_clasification):
                    text = poly_data.ocr_text
                    logger.info(f"{poly}: '{text}', clas: {poly_data.semantic_clasification} | t_cuant: {poly_data.cuant_chars} / len: {len("".join(text))}")
        return True
        
    def classify_strings(self, manager: DataFormatter) -> bool:
        try:
            if not manager.workflow or not manager.workflow.polygons:
                logger.warning("Semantic Clasificator no tiene polígonos para procesar")
                return False
                
            polygons_to_classify: Dict[str, Polygons] = manager.workflow.polygons
            
            if not polygons_to_classify:
                logger.warning("No hay polígonos que clasificar")
                return False
            
            # Clasificar solo los polígonos seleccionados
            # t0 = time.perf_counter()
            final_results: Dict[str, Tuple[List[int], int]] = clasify_words(polygons_to_classify)
            # logger.info(f"Tiempo de clasificación: {time.perf_counter() - t0:.6f}'s")
            manager.update_semantic_clasification(final_results)
            return True

        except Exception as e:
            logger.warning(f"Error en el clasificador: {e}", exc_info=True)
            return False
            
    def get_early_data(self, manager: DataFormatter) -> bool:
        """Asigna key_field sobre texto OCR crudo antes de fragmentar/clasificar. Cada tipo de dato se marca como mucho una vez por documento (orden lectura: poly_index)."""
        if not manager.workflow or not manager.workflow.polygons:
            logger.warning("Semantic Clasificator no tiene polígonos para procesar")
            return False

        polygons: Dict[str, Polygons] = manager.workflow.polygons
        # [fecha, rfc, iva] — ya satisfechos en el documento
        state: List[bool] = [False, False, False, False, False, False]

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
            manager.update_key_field(polygon_updates)    
            return True
            
        return False
            
    def preprocess_text(self, manager: DataFormatter) -> bool:
        if not manager.workflow or not manager.workflow.polygons:
            logger.warning("Semantic Clasificator no tiene polígonos para procesar")
            return False
            
        polygons: Dict[str, Polygons] = manager.workflow.polygons
        final_polygons: Dict[str, Dict[str, Any]] = {}
        
        for poly, poly_data in polygons.items():
            text = poly_data.ocr_text or ""
            if len(text) < 2:
                final_polygons[poly] = {"text": text}
                continue
            
            elif text.isalpha() or text.isdecimal():
                final_polygons[poly] = {"text": text}
                continue
                
            elif contains_quantitative(text) and not is_quantitative(text):
                qtext = get_cuants(text)
                # logger.info(f"POTENCIAL CUANTS: '{text}'")
                if qtext != text:
                    # logger.info(f"CUANT ENCONTRADO: '{poly}' | Texy: '{text}' -> '{set(text.split(" ")).difference(set(qtext.split(" ")))}' → '{qtext}'")
                    final_polygons[poly] = {"text": qtext}
                    continue

                final_polygons[poly] = {"text": qtext}
                continue
                    
            if contains_umd(text):
                umd_text = find_umd(text)
                # logger.info(f"POTENCIAL UMDS: '{text}'")
                if umd_text != text:
                    # logger.info(f"UMD ENCONTRADA:'{poly}' | Text: '{text}' -> '{set(text.split(" ")).difference(set(umd_text.split(" ")))}' → '{umd_text}'")
                    final_polygons[poly] = {"text": umd_text}
                    continue
                
                final_polygons[poly] = {"text": umd_text}
                continue
            else:
                final_polygons[poly] = {"text": text}
                continue

        return manager.update_ocr_results(final_polygons, "refiner")