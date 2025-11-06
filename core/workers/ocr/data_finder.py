import time
from typing import Dict, Any, Optional, List, Tuple
import logging
from core.domain.data_models import Polygons
from core.factory.abstract_worker import OCRAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.models_manager import ModelsManager
from core.utils.text_validator import validate_text
from core.utils.pattern_finder import find_rfc, find_iva, find_date

logger = logging.getLogger(__name__)

class DataFinder(OCRAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.config=config
        self.worker_config = self.config.get('data_finder', {})
        self.threshold = float(self.worker_config.get("min_similarity"))
        self.max_q_lenght: Tuple[int, int] = self.worker_config["max_q_lenght"]
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
                logger.debug(f"Key Fields detectados en {time.perf_counter() - start_time:.6f}s")
                return True
                
        except Exception as e:
            logger.error(f"Error detectando encabezados por palabra: {e}", exc_info=True)
        return True

    def _find_data(self, polygons: Dict[str, Polygons]) -> Dict[str, int]:
        time0 = time.perf_counter()
        if self.model is None:
            logger.error("DataFinder no iniciado, no se puede búsacar texto")
            return {}

        try:
            processed_count = 0
            polygon_updates: Dict[str, int] = {}
            skipped_semantic = 0
            skipped_len = 0

            for pid, poly in polygons.items():
                processed_count += 1

                sc = poly.semantic_clasification
                if sc == 1 or sc == 2 or sc == -1 or sc == -2:
                    skipped_semantic += 1
                    continue

                ocr_text = poly.ocr_text or ""
                word_lenght = len(ocr_text)
                if not validate_text(ocr_text) or word_lenght < self.max_q_lenght[0] or word_lenght > self.max_q_lenght[1]:
                    logger.debug(f"Polygono {pid} sin texto o excede longitud: '{ocr_text}', letras: '{word_lenght}'")
                    skipped_len += 1
                    continue

                rfc_key = find_rfc(ocr_text)
                if rfc_key:
                    skipped_semantic +=1
                    logger.warning(f"RFC encontrado en {pid}, '{ocr_text}', {rfc_key}")
                    polygon_updates[pid] = 7
                    continue

                iva_key = find_iva(ocr_text)
                if iva_key:
                    skipped_semantic +=1
                    logger.warning(f"IVA encontrado en {pid}, '{ocr_text}', {iva_key}")
                    polygon_updates[pid] = 8
                    continue

                date_key = find_date(ocr_text)
                if date_key:
                    skipped_semantic +=1
                    logger.warning(f"FECHA encontrado en {pid}, '{ocr_text}', {date_key}")
                    polygon_updates[pid] = 9
                    continue

                valid_results: List[Dict[str, Any]] = self.model.find_keywords(ocr_text, self.threshold)
                if not valid_results:
                    continue

                if valid_results:
                    best_result: Dict[str, Any] = max(valid_results, key=lambda x: x['similarity'])
                    key_field: int = best_result['key_field']

                    if key_field:
                        polygon_updates[pid] = key_field
                        logger.debug(f"Resultado de {pid}: {best_result}")

            if polygon_updates:
                logger.debug(f"{skipped_semantic} polígonos semánticos omitidos")
                logger.debug(f"Encontradas {len(polygon_updates)} coincidencias en {time.perf_counter() - time0:6f}s")
                return polygon_updates

            else:
                logger.warning("No se encontraron coincidencias de palabras clave")
                return {}
                    
        except Exception as e:
            logger.warning(msg=f"Fallo en búsqueda de datos globales: {e}", exc_info=True)
            return {}
