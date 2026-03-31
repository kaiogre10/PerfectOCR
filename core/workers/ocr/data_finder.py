# core/workers/ocr/data_finder.py
import time
from typing import Dict, Any, Optional, List
import logging
from core.domain.data_models import Polygons
from core.factory.abstract_worker import OCRAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.models_manager import ModelsManager
from core.utils.text_utils import find_rfc, find_iva, find_date

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

        except Exception as e:
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
            found_date = False
            found_rfc = False
            found_iva = False

            for pid, poly in polygons.items():
                processed_count += 1

                ocr_text = poly.ocr_text or ""
                text_len = len(ocr_text)
                sc = poly.semantic_clasification
                set_sc = set(sc)
                sc_forb = {-2, 1, 2}
                if not ocr_text:
                    skipped_semantic += 1
                    continue

                elif text_len == 1:
                    skipped_semantic += 1
                    # logger.info(f"Skipeado por longitud: {ocr_text}")
                    continue

                elif len(sc) == 1 and not sc_forb.isdisjoint(set_sc):
                    # logger.info(f"Skipeado por sc: {ocr_text}: {sc}")
                    skipped_semantic += 1
                    continue

                elif not found_date and find_date(ocr_text):
                    skipped_semantic +=1
                    # logger.info(f"FECHA encontrado en {pid}, '{ocr_text}', sc: {poly.semantic_clasification}")
                    found_date = True
                    polygon_updates[pid] = 9
                    continue

                elif not found_rfc and text_len > 11 and find_rfc(ocr_text):
                    skipped_semantic +=1
                    found_rfc = True
                    # logger.debug(f"RFC encontrado en {pid}, '{ocr_text}'")
                    polygon_updates[pid] = 7
                    continue

                elif not found_iva and find_iva(ocr_text):
                    skipped_semantic +=1
                    found_iva = True
                    # logger.info(f"IVA encontrado en {pid}, '{ocr_text}'")
                    polygon_updates[pid] = 8
                    continue

                else:
                    ocr_text = ocr_text.lower()
                    # logger.info(f"Poly: {pid}: TEXTO: '{ocr_text}")
                    valid_results: List[Dict[str, Any]] = self.model.find_keywords(ocr_text)
                    if not valid_results:
                        continue

                    # logger.info(f"Results: {valid_results}")
                    num_keywords = len(valid_results)
                    all_key_fields = [result['key_field'] for result in valid_results]

                    # Verificar si todos son headers (key_field == 6)
                    if num_keywords > 1 and all(kf == 6 for kf in all_key_fields):
                        polygon_updates[pid] = all_key_fields
                        logger.debug(f"'{len(all_key_fields)}': {all_key_fields} headers en {pid}")

                    else:
                        key_field = valid_results[0]['key_field']
                        polygon_updates[pid] = key_field
                        logger.debug(f"'{pid}': Key_Field {key_field}")

            if polygon_updates:
                # logger.info(f"KEY_FIELDS: {polygon_updates}")
                # logger.info(f"Cantidad de keyfields: {len(polygon_updates)} completados en: {time.perf_counter() - time0:.6}, {skipped_semantic} omisiones")
                return polygon_updates

            else:
                logger.warning("No se hallaron Keywords")
                return {}

        except Exception as e:
            logger.warning(f"Error encontrando keyfields: {e}")
        return {}
    