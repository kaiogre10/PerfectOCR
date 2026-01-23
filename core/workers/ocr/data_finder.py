# core/workers/ocr/data_finder.py
import time
from typing import Dict, Any, Optional, List
import logging
from core.domain.data_models import Polygons
from core.factory.abstract_worker import OCRAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.models_manager import ModelsManager
from core.utils.text_validator import validate_text, estandarice_uppers_lowers
from core.utils.pattern_finder import find_rfc, find_iva, find_date, find_umd

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
                logger.warning("Modelo de búsqueda obtenido del ModelsManager")
            return self._model #type: ignore

        except Exception as e:
            logger.error(f"DataFinder: Modelo de búsqueda no disponible en ModelManager{e}", exc_info=True)
            return None

    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        logger.debug("Data Finder iniciado")
        start_time = time.perf_counter()
        try:
            polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
            if not polygons:
                logger.error("No hay polygons para procesar")
                return False
            
            # Llamar al meetodo original que funciona
            polygon_updates = self._find_data(polygons)

            # Actualiza los key_fields
            if manager.update_key_field(polygon_updates):
                logger.info(f"Key Fields detectados en {time.perf_counter() - start_time:.6f}s")
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
            skipped_len = 0

            for pid, poly in polygons.items():
                processed_count += 1

                sc = poly.semantic_clasification
                if sc == 1 or sc == 2 or sc == -1 or sc == -2:
                    logger.debug(f"{pid} omitido semanticamente sc= '{sc}': ")
                    skipped_semantic += 1
                    continue

                ocr_text = poly.ocr_text or ""
                word_lenght = len(ocr_text)
                if not validate_text(ocr_text) or word_lenght < 2:
                    logger.info(f"{pid} sin texto o excede longitud: '{ocr_text}', letras: '{word_lenght}'")
                    skipped_len += 1
                    continue

                if find_umd(ocr_text):
                    skipped_semantic += 1
                    logger.info(f"'{pid}' UMD: {ocr_text}")
                    continue

                date_key = find_date(ocr_text)
                if date_key:
                    skipped_semantic +=1
                    logger.info(f"FECHA encontrado en {pid}, '{ocr_text}'")
                    polygon_updates[pid] = 9
                    continue

                rfc_key = find_rfc(ocr_text)
                if rfc_key:
                    skipped_semantic +=1
                    logger.info(f"RFC encontrado en {pid}, '{ocr_text}'")
                    polygon_updates[pid] = 7
                    continue

                iva_key = find_iva(ocr_text)
                if iva_key:
                    skipped_semantic +=1
                    logger.info(f"IVA encontrado en {pid}, '{ocr_text}'")
                    polygon_updates[pid] = 8
                    continue
                
                valid_results: List[Dict[str, Any]] = self.model.find_keywords(ocr_text)
                if valid_results:
                    # continue

                    num_keywords = len(valid_results)
                    all_key_fields = [result['key_field'] for result in valid_results]
                    
                    # Verificar si todos son headers (key_field == 6)
                    if num_keywords > 1 and all(kf == 6 for kf in all_key_fields):
                        # Múltiples headers: asignar como lista
                        polygon_updates[pid] = all_key_fields
                        pot_headers = " ".join(result["key_word"] for result in valid_results)
                        head_standar = estandarice_uppers_lowers(ocr_text, pot_headers)
                        poly.ocr_text = head_standar
                        logger.debug(f"'{len(all_key_fields)}': {all_key_fields} headers en {pid}")

                    else:
                        key_field = valid_results[0]['key_field']
                        polygon_updates[pid] = key_field
                        logger.debug(f"'{pid}': Key_Field {key_field}")

                    continue

                # logger.info(f"{pid}: exto superviviente {ocr_text}")

            if polygon_updates:
                logger.debug(f"KEY_FIELDS: {polygon_updates}")
                logger.debug(f"Cantidad de keyfields: {len(polygon_updates)} completados en: {time.perf_counter() - time0:.6}")
                return polygon_updates
            
            else:
                logger.warning("No se hallaron Keywords")
                return {}
        
        except Exception as e:
            logger.warning(f"Error encontrando keyfields: {e}")
        return {}
                    