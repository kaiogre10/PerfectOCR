import time
from typing import Dict, Any, Optional, List
import logging
import re
from cleantext import clean #type: ignore
from core.domain.data_models import Polygons
from core.factory.abstract_worker import OCRAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.models_manager import ModelsManager
from fuzzywuzzy import utils

logger = logging.getLogger(__name__)

class DataFinder(OCRAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.config=config
        self.worker_config = self.config.get('data_finder', {})
        self._model = None

    @property
    def model(self) -> Optional[Any]:
        try:
            if self._model is None: #type: ignore
                model_manager = ModelsManager.get_instance()
                self._model = model_manager.word_finder #type: ignore
                logger.debug("DataFinder: Modelo de búsqueda obtenido del ModelsManager")
            return self._model #type: ignore

        except Exception as e:
            logger.error(f"DataFinder: Modelo de búsqueda no disponible en ModelManager{e}", exc_info=True)
            return None

    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:

        start_time = time.time()
        try:
            logger.debug("Data Finder iniciado")
            if not manager or not getattr(manager, "workflow", None):
                logger.warning("Manager o workflow ausente")
                return False
            
            workflow = manager.workflow
            polygons: Dict[str, Polygons] = getattr(workflow, "polygons", {}) or {}
                
            if not polygons:
                logger.error("No hay polygons para procesar")
                return False
            
            # Llamar al método original que funciona
            polygon_updates = self._find_data(polygons)

            # Actualiza las líneas marcadas como encabezado en las dataclasses
            if manager.update_key_field(polygon_updates):
                total_time = time.time() - start_time
                logger.debug(f"Key Fields detectados en {total_time:6f}s")
                return True
                
        except Exception as e:
            logger.error(f"Error detectando encabezados por palabra: {e}", exc_info=True)
        return True

    def _find_data(self, polygons: Dict[str, Polygons]) -> Dict[str, str]:
        time0 = time.perf_counter()
        threshold = float(self.worker_config.get("min_similarity"))

        max_q_lenght = int(self.worker_config.get("max_q_lenght"))
        
        if self.model is None:
            logger.error("DataFinder no iniciado, no se puede búsacar texto")
            return {}

        try:
            if not polygons:
                logger.error("No hay polígonos para procesar")
                return {}

            processed_count = 0
            polygon_updates: Dict[str, str] = {}
            skipped_numeric = 0
            skipped_len = 0

            for pid, poly in polygons.items():
                # Obtener datos del polígono
                processed_count += 1
                ocr_text: str = poly.ocr_text
                sc = poly.semantic_clasification

                # Validación del texto antes de procesar
                if not utils.validate_string(ocr_text):
                    logger.info(msg=f"Polygono sin texto: {pid}")
                    continue

                if sc:
                    if sc.numeric or sc.quantitative or sc.code or sc.rfc:
                        skipped_numeric += 1
                    continue

                lenght = len(ocr_text)

                if max_q_lenght is not None and lenght > max_q_lenght:
                    skipped_len += 1
                    logger.info(f"{pid}, texto: '{ocr_text}' omitido por largo ({lenght} > {max_q_lenght})")
                    continue

                valid_results: List[Dict[str, Any]] = self.model.find_keywords(ocr_text, threshold)
                if not valid_results:
                    continue

                if valid_results:
                    best_result: Dict[str, Any] = max(valid_results, key=lambda x: x['similarity'])
                    key_field: str = best_result['key_field']

                    if key_field:
                        polygon_updates[pid] = key_field
                        logger.info(f"Resultado de {pid}: {best_result}")

            if polygon_updates:
                logger.info(f"{skipped_numeric} polígonos omitidos")
                logger.info(f"Encontradas {len(polygon_updates)} coincidencias en {time.perf_counter() - time0:6f}s")
                return polygon_updates

            else:
                logger.warning("No se encontraron coincidencias de palabras clave")
                return {}
                    
        except Exception as e:
            logger.warning(msg=f"Fallo en búsqueda de datos globales: {e}", exc_info=True)
            return {}

    def _find_rfc(self, s: str) -> bool:

        try:

            if not s or not s.strip():
                return False

            rfc_code = r'^([A-ZÑ&]{3,4})\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])[A-Z0-9]{3}$'
            rfc_word = r'\b(R\.?F\.?C\.?)\b'

            # Busca primero el patrón corto
            if re.search(rfc_word, s):
                # Si lo encuentra, busca el patrón largo

                if re.search(rfc_code, s):
                    logger.info(f"Resultado de RFC: {s}")
                    return True

                else:
                    return False

            # Si no encuentra el corto, busca el largo directamente
            if re.search(rfc_code, s):
                logger.info(f"Resultado de RFC: {s}")
                return True

            return False

        except Exception as e:
            logger.info(f"Error buscando RFC: {e}", exc_info=True)
            return False
