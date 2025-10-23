import time
from typing import Dict, Any, Optional, List
import logging
from core.domain.data_models import Polygons
from core.factory.abstract_worker import OCRAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.models_manager import ModelsManager
from fuzzywuzzy import utils # type: ignore
from core.utils.pattern_finder import find_rfc, find_iva, find_date

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

            polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
                
            if not polygons:
                logger.error("No hay polygons para procesar")
                return False

            if manager.create_semantic_clasification():
                logger.debug("Clasificación semántica creada")
            
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
            processed_count = 0
            polygon_updates: Dict[str, str] = {}
            skipped_semantic = 0
            skipped_len = 0

            for pid, poly in polygons.items():
                # Obtener datos del polígono
                processed_count += 1
                ocr_text = poly.ocr_text or ""

                # Validación del texto antes de procesar
                if not utils.validate_string(ocr_text): #type: ignore
                    logger.debug(msg=f"Polygono sin texto: {pid}")
                    continue

                sc = poly.semantic_clasification
                if sc.numeric or sc.quantitative or sc.code:
                    skipped_semantic += 1
                    continue

                lenght = len(ocr_text)

                if lenght > max_q_lenght:
                    skipped_len += 1
                    logger.debug(f"{pid}, texto: '{ocr_text}' omitido por largo ({lenght} > {max_q_lenght})")
                    continue

                rfc_key = find_rfc(ocr_text)
                if rfc_key:
                    skipped_semantic +=1
                    logger.info(f"RFC encontrado en {pid}, '{ocr_text}', {rfc_key}")
                    polygon_updates[pid] = "RFCProveedor"
                    continue

                iva_key = find_iva(ocr_text)
                if iva_key:
                    skipped_semantic +=1
                    logger.info(f"IVA encontrado en {pid}, '{ocr_text}', {iva_key}")
                    polygon_updates[pid] = "MontoIVAGeneral"
                    continue

                date_key = find_date(ocr_text)
                if date_key:
                    skipped_semantic +=1
                    logger.info(f"FECHA encontrado en {pid}, '{ocr_text}', {date_key}")
                    polygon_updates[pid] = "FechaDocumento"
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
                logger.debug(f"{skipped_semantic} polígonos semánticos omitidos")
                logger.debug(f"Encontradas {len(polygon_updates)} coincidencias en {time.perf_counter() - time0:6f}s")
                return polygon_updates

            else:
                logger.warning("No se encontraron coincidencias de palabras clave")
                return {}
                    
        except Exception as e:
            logger.warning(msg=f"Fallo en búsqueda de datos globales: {e}", exc_info=True)
            return {}
    