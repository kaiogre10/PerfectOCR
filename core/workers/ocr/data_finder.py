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

        except ImportError as e:
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

    def _find_data(self, polygons: Dict[str, Polygons]) -> Dict[str, List[int] | int]:
        time0 = time.perf_counter()
        if self.model is None:
            logger.error("DataFinder no iniciado, no se puede búsacar texto")
            return {}

        try:
            processed_count = 0
            polygon_updates: Dict[str, List[int] | int] = {}
            skipped_semantic = 0
            # found_date = False
            # found_rfc = False
            # found_iva = False
            sc_forb = {2, 1, -2}

            all_idx = np.array([p.poly_index for p in polygons.values()], np.int16)

            sc = [p.semantic_clasification for p in polygons.values()]
            texts = [(p.ocr_text or "") for p in polygons.values()]

            texts_length = np.array([len(t) for t in texts])
            decimal_p = np.array([t.isdecimal() for t in texts])

            sc_length = np.array([len(c) for c in sc])
            forb_sc = np.array([any(c in sc_forb for c in s) for s in sc])

            mask_sc = (sc_length == 1) & (forb_sc == True) 
            mask_len = (texts_length < 2) & (decimal_p == True)
            mask = mask_sc | mask_len
            skip_idx = np.compress(mask, all_idx).tolist()

            for pid, poly in polygons.items():
                if poly.poly_index in skip_idx:
                    # logger.info(f"{pid} Omitido: '{poly.ocr_text}' | sc: {poly.semantic_clasification}")
                    skipped_semantic += 1
                    continue

                processed_count += 1

                ocr_text = poly.ocr_text or ""
                # logger.info(f"Texto a procesar: {ocr_text}")
                if ocr_text.isdecimal():
                    skipped_semantic += 1
                    continue

                if not validate_text(ocr_text):
                    skipped_semantic += 1
                    continue

                # if not found_date and self.find_date(ocr_text):
                #     skipped_semantic +=1
                #     logger.info(f"FECHA encontrado en {pid}, '{ocr_text}'")
                #     found_date = True
                #     polygon_updates[pid] = 9
                #     continue

                # if not found_rfc and self.find_rfc(ocr_text):
                #     skipped_semantic +=1
                #     found_rfc = True
                #     # logger.info(f"RFC encontrado en {pid}, '{ocr_text}'")
                #     polygon_updates[pid] = 7
                #     continue

                # elif not found_iva and self.find_iva(ocr_text):
                #     skipped_semantic +=1
                #     found_iva = True
                #     # logger.info(f"IVA encontrado en {pid}, '{ocr_text}'")
                #     polygon_updates[pid] = 8
                #     continue

                else:
                    ocr_text = ocr_text.lower()
                    #logger.info(f"Poly: {pid}: TEXTO: '{ocr_text}")
                    valid_results: List[Dict[str, Any]] = self.model.find_keywords(ocr_text)
                    if not valid_results:
                        continue

                    # logger.info(f"Results: {valid_results}")
                    num_keywords = len(valid_results)
                    all_key_fields = [result['key_field'] for result in valid_results]

                    # Verificar si todos son headers (key_field == 6)
                    if num_keywords > 1 and all(kf == 6 for kf in all_key_fields):
                        polygon_updates[pid] = all_key_fields
                        logger.info(f"'{len(all_key_fields)}': {all_key_fields} headers en {pid}")

                    else:
                        key_field = valid_results[0]['key_field']
                        polygon_updates[pid] = key_field
                        # logger.info(f"'{pid}': Key_Field: '{key_field}'")

            if polygon_updates:
                # logger.info(f"KEY_FIELDS: {polygon_updates}")
                logger.info(f"Cantidad de keyfields: {len(polygon_updates)} completados en: {time.perf_counter() - time0:.6}, {skipped_semantic} omisiones")
                return polygon_updates

            else:
                logger.warning("No se hallaron Keywords")
                return {}

        except Exception as e:
            logger.warning(f"Error encontrando keyfields: {e}")
        return {}
    
    # def find_date(self, texts: List[str]) -> Optional[List[int]]:
    #     """
    #     Busca la primera coincidencia de fecha en la lista completa de textos.
    #     Devuelve los índices de los polígonos que componen la fecha.
    #     """
    #     try:
    #         # Usar un separador que no interfiera con los patrones de fecha
    #         separator = " "
    #         full_text = separator.join(texts)

    #         logger.info(f"texts: {texts}")
            
    #         match = DATE_PATTENRS.findall(full_text)
    #         if not match:
    #             return None

    #         # Calcular los índices de los polígonos que componen la fecha
    #         start_char, end_char = match.span()
            
    #         # Contar cuántos espacios (separadores) hay antes del inicio del match
    #         # para saber en qué índice de la lista original de textos empezar.
    #         start_poly_index = full_text[:start_char].count(separator)
            
    #         # Contar cuántos polígonos abarca el texto encontrado
    #         matched_text = full_text[start_char:end_char]
    #         num_polys_in_match = matched_text.count(separator) + 1
            
    #         return list(range(start_poly_index, start_poly_index + num_polys_in_match))

    #     except Exception as e:
    #         logger.warning(f"Error buscando fecha global: {e}", exc_info=True)
    #         return None
# def find_date(self, s: str) -> bool:
#         try:
#             if s.isalpha():
#                 return False
#             else:
#                 return bool(DATE_PATTENRS.search(s))

#         except TypeError as e:
#             logger.warning(f"Error buscando fecha: {e}", exc_info=True)
#         return False
