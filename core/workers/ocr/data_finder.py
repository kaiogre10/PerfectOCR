# core/workers/ocr/data_finder.py
import time
from typing import Dict, Any, Optional, List
import logging
import numpy as np
from core.domain.data_models import Polygons
from core.factory.abstract_worker import OCRAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.models_manager import ModelsManager
from core.utils.text_utils import validate_text

logger = logging.getLogger(__name__)

class DataFinder(OCRAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self._model = None

    @property
    def model(self) -> Optional[Any]:
        try:
            if self._model is None: #type: ignore
                model_manager = ModelsManager.get_instance()
                self._model = model_manager.word_finder #type: ignore
                logger.debug("Modelo de búsqueda obtenido del ModelsManager")
            return self._model #type: ignore

        except ModuleNotFoundError as e:
            logger.error(f"DataFinder: Modelo de búsqueda no disponible en ModelManager{e}", exc_info=True)
        return None

    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        logger.debug("Data Finder iniciado")
        try:
            polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
            if not polygons:
                logger.error("No hay polygons para procesar")
                return False

            polygon_updates = self._find_data(polygons)
            if manager.update_key_field(polygon_updates):
                return True
                
        except Exception as e:
            logger.error(f"Error detectando encabezados por palabra: {e}", exc_info=True)
        return True

    def _find_data(self, polygons: Dict[str, Polygons]) -> Dict[str, List[int]]:
        if self.model is None:
            logger.error("DataFinder no iniciado, no se puede buscar Key FIelds")
            return {}
        time0 = time.perf_counter()
        try:
            processed_count = 0
            polygon_updates: Dict[str, List[int]] = {}
            skipped_semantic = 0
            sc_forb = {0, 2, 4, 5}

            all_idx = np.array([p.poly_index for p in polygons.values()], np.int16)

            sc = [p.semantic_clasification for p in polygons.values()]
            texts = [(p.ocr_text or "") for p in polygons.values()]

            texts_length = np.array([len(t) for t in texts])
            decimal_p = np.array([t.isnumeric() for t in texts])

            # sc_length = np.array([len(c) for c in sc])
            sc_eq = np.array([len(set(t)) for t in sc])
            forb_sc = np.array([any(c in sc_forb for c in s) for s in sc])

            mask_sc = (sc_eq == 1) & (forb_sc == True) 
            mask_len = (texts_length < 2) & (decimal_p == True)
            mask = mask_sc | mask_len
            skip_idx = np.compress(mask, all_idx).tolist()

            for pid, poly in polygons.items():
                if poly.poly_index in skip_idx:
                    # logger.info(f"{pid} Omitido: '{poly.ocr_text}' | sc: {poly.semantic_clasification}")
                    skipped_semantic += 1
                    continue

                processed_count += 1
                kf = poly.key_field or None
                if kf is not None or kf:
                    skipped_semantic += 1
                    # logger.info(f"KeyField redundante en WODR FINDER {pid}: '{poly.ocr_text}' | sc: {poly.semantic_clasification}")
                    continue

                ocr_text = poly.ocr_text or ""

                if not validate_text(ocr_text):
                    # logger.info(f"Texto INVÁLIDO: '{ocr_text}' | sc: {poly.semantic_clasification}")
                    skipped_semantic += 1
                    continue
            
                else:
                    # if is_acronym(ocr_text):
                    #     ocr_text = ocr_text.replace(".", "")
                    # logger.info(f"Texto a procesar: '{ocr_text}' | sc: {poly.semantic_clasification}")
                    valid_results: List[Dict[str, Any]] = self.model.find_keywords(ocr_text)
                    if not valid_results:
                        continue

                    # logger.info(f"Valid Results: {valid_results}")
                    num_keywords = len(valid_results)
                    all_key_fields = [result['key_field'] for result in valid_results]
                    if not all_key_fields or all_key_fields is None:
                        continue

                    # Verificar si todos son headers (key_field == 6)
                    if all(kf == 6 for kf in all_key_fields):
                    
                        if num_keywords > 1:
                            polygon_updates[pid] = all_key_fields
                            # logger.info(f"'{len(all_key_fields)}': {all_key_fields} headers en {pid}")
                            continue
                    
                        key_word = set([result['key_word'] for result in valid_results])
                        orig_text = set([result['norm_ocr_text'] for result in valid_results])
                        leftovers = orig_text.difference(key_word)
                        if leftovers:
                            add_kf = len(leftovers)
                            # logger.info(f"KW: '{key_word}' | ORIG: '{orig_text}' -> '{leftovers}'")
                            # Asigna una lista de 6's: uno por cada palabra (keyword + leftovers)
                            polygon_updates[pid] = [6] * (add_kf + 1)
                            continue
                        
                        key_field = valid_results[0]['key_field']
                        polygon_updates[pid] = [key_field]
                        continue
                    
                    else:
                        key_field = valid_results[0]['key_field']
                        polygon_updates[pid] = [key_field]
                        # logger.info(f"'{pid}': Key_Field: '{key_field}', Text: '{ocr_text}'")
                        
            if polygon_updates:
                # logger.info(f"KEY_FIELDS: {polygon_updates}")
                # logger.info(f"KEY FIELDS ENCONTRADOS: '{len(polygon_updates)}', en: {time.perf_counter() - time0:.6}'s, {skipped_semantic} omisiones")
                return polygon_updates

            else:
                logger.warning(f"No se hallaron Keywords, tiempo de ejecución: {time.perf_counter() - time0:.6}'s")
                return {}

        except ValueError as e:
            logger.warning(f"Error encontrando keyfields: {e}")
        return {}