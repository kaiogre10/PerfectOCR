# core/workers/ocr/text_refiner.py
from typing import Dict, Any, Optional, List, Tuple
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons
from core.factory.abstract_worker import OCRAbstractWorker
from core.workers.ocr.text_cleaner import TextCleaner
from core.workers.ocr.fragmenter import Fragmenter
from services.output_service import save_raw_json
from core.utils.text_utils import clasify_words, get_cuants, contains_quantitative, find_key_data
import logging
import dataclasses
import time

logger = logging.getLogger(__name__)

class Refiner(OCRAbstractWorker):
    """
    Orquesta un ciclo de refinamiento de texto post-OCR con clasificación selectiva optimizada.
    """
    def __init__(self, config: Dict[str, Any], project_root: str, cleaner: Optional[TextCleaner] = None,  fragmenter: Optional[Fragmenter] = None):
        super().__init__(config, project_root)
        self.worker_config = config.get("text_refiner", {})
        self.num_passes = self.worker_config.get("num_passes")
        self.output = config.get("cleanned_text")
        self.cleaner = cleaner
        self.fragmenter = fragmenter

    def _log_worker_time(self, pass_num: int, worker_name: str, start_time: float, stage_name: str = "") -> None:
        elapsed = time.perf_counter() - start_time
        stage_label = f" | Etapa: {stage_name}" if stage_name else ""
        logger.debug(f"Bucle #{pass_num} | Worker: {worker_name}{stage_label} | Tiempo: {elapsed:.6f}s")

    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """
        Ejecuta el ciclo de refinamiento con clasificación selectiva.
        """
        t0 = time.perf_counter()
        self.preprocess_text(manager)
        self.get_early_data(manager)
        try:
            if 0 >= self.num_passes:
                # step_t0 = time.perf_counter()
                self.classify_strings(manager)
                # self._log_worker_time(0, "SemanticClassifier", step_t0, "Clasificación Semántica")
            
            else:
                for i in range(self.num_passes):
                    pass_num = i + 1
                    # pass_t0 = time.perf_counter()
                    # logger.debug(f"Iniciando Bucle de Refinamiento de Texto #{pass_num}")

                    # logger.debug(f"Pasada 1, bucle #{pass_num}: Clasificación Semántica")
                    # step_t0 = time.perf_counter()
                    # self.classify_strings(manager)
                    # self._log_worker_time(pass_num, "SemanticClassifier", step_t0, "Clasificación Semántica")
                    
                    if self.cleaner:
                    #     cleaner_name = self.cleaner.__class__.__name__
                        # logger.info(f"Bucle #{pass_num}: Limpieza de Texto")
                    #     step_t0 = time.perf_counter()
                        self.cleaner.transcribe(context, manager)
                        # self._log_worker_time(pass_num, cleaner_name, step_t0, "Limpieza de Texto")
                        
                    # logger.info(f"Pasada 1, bucle #{pass_num}: Clasificación Semántica (solo limpios)")
                    # step_t0 = time.perf_counter()
                    self.classify_strings(manager)
                    # self._log_worker_time(pass_num, "SemanticClassifier", step_t0, "Clasificación Semántica (solo limpiados)")

                    if self.fragmenter:
                        # fragmenter_name = self.fragmenter.__class__.__name__
                        # logger.info(f"Bucle #{pass_num}: Fragmentación Textual")
                        # step_t0 = time.perf_counter()
                        self.fragmenter.transcribe(context, manager)
                        # self._log_worker_time(pass_num, fragmenter_name, step_t0, "Fragmentación Textual")

                    # logger.debug(f"Pasada 2, bucle #{pass_num}: Clasificación Semántica (solo fragmentados)")
                    # # step_t0 = time.perf_counter()
                    # self.classify_strings(manager)
                    # self._log_worker_time(pass_num, "SemanticClassifier", step_t0, "Clasificación Semántica (solo fragmentados)")
                    
                    # if self.corrector:
                    #     corrector_name = self.corrector.__class__.__name__
                    #     logger.debug(f"Bucle #{pass_num}: Corrección textual")
                    #     step_t0 = time.perf_counter()
                    #     self.corrector.transcribe(context, manager)
                    #     self._log_worker_time(pass_num, corrector_name, step_t0, "Corrección textual")

                    # logger.debug(f"Bucle #{pass_num} | Tiempo total iteración: {time.perf_counter() - pass_t0:.6f}s")
                # logger.debug(f"Pasada final: Clasificación Semántica completa")
                # step_t0 = time.perf_counter()
                self.classify_strings(manager)
                # self._log_worker_time(self.num_passes + 1, "SemanticClassifier", step_t0, "Clasificación Semántica final")
                
            logger.info(f"Tiempo de refinado: {time.perf_counter() - t0:.6f}'s")
            polygons = manager.workflow.polygons if manager.workflow else {}
            for poly, poly_data in polygons.items():
                logger.debug(f"{poly}: '{poly_data.ocr_text}', clas: {poly_data.semantic_clasification}")
                # if any(sc in poly_data.semantic_clasification for sc in (-1, 0)):
                #     logger.info(f"{poly}: '{poly_data.ocr_text}', clas: {poly_data.semantic_clasification}")
            
            if self.output:
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
                return False
                                        
            # Clasificar solo los polígonos seleccionados
            # t0 = time.perf_counter()
            final_results: Dict[str, Tuple[List[int], int]] = clasify_words(polygons_to_classify, self.worker_config)
            # logger.info(f"Tiempo de clasificación: {time.perf_counter() - t0:.6f}'s")

            manager.update_semantic_clasification(final_results)

            return True

        except Exception as e:
            logger.warning(f"Error en el clasificador: {e}", exc_info=True)
            return False
            
    def get_early_data(self, manager: DataFormatter) -> bool:
        """
        Asigna key_field (fecha, RFC, IVA) sobre texto OCR crudo antes de fragmentar/clasificar.
        Cada tipo de dato se marca como mucho una vez por documento (orden lectura: poly_index).
        """
        try:
            if not manager.workflow or not manager.workflow.polygons:
                logger.warning("Semantic Clasificator no tiene polígonos para procesar")
                return False

            polygons: Dict[str, Polygons] = manager.workflow.polygons
            # [fecha, rfc, iva] — ya satisfechos en el documento
            state: List[bool] = [False, False, False, False, False]

            for _, pd in polygons.items():
                kf = pd.key_field
                if kf is None:
                    continue
                keys = kf if isinstance(kf, list) else [kf]
                if 9 in keys:
                    state[0] = True
                if 7 in keys:
                    state[1] = True
                if 8 in keys:
                    state[2] = True
                if 10 in keys:
                    state[3] = True
                if 11 in keys:
                    state[4] = True

            polygon_updates: Dict[str, List[int] | int] = {}

            for poly_id, poly_data in polygons.items():
                if poly_data.key_field is not None:
                    continue

                text = poly_data.ocr_text or ""
                key_field = find_key_data(text, state)

                if key_field is None:
                    continue
                
                polygon_updates[poly_id] = key_field

            if polygon_updates:
                manager.update_key_field(polygon_updates)
                
                return True
            return False
        except Exception as e:
            logger.warning(f"Error encontrando keydata: '{e}", exc_info=True)
        return False
            
    def preprocess_text(self, manager: DataFormatter) -> bool:
        try:
            if not manager.workflow or not manager.workflow.polygons:
                logger.warning("Semantic Clasificator no tiene polígonos para procesar")
                return False
                
            polygons: Dict[str, Polygons] = manager.workflow.polygons
            updated_polygons: Dict[str, Polygons] = {}
            for poly, poly_data in polygons.items():
                text = poly_data.ocr_text or ""
                if contains_quantitative(text):
                    qtext = get_cuants(text)
                    if qtext != text:
                        # logger.info(f"CUANTS: '{text}' -> '{qtext}'")
                        updated_polygons[poly] = dataclasses.replace(poly_data, ocr_text=qtext)
                    else:
                        updated_polygons[poly] = poly_data
                else:
                    updated_polygons[poly] = poly_data

            manager.workflow.polygons = updated_polygons
        except Exception as e:
            logger.warning(f"Error preprocesando texto: '{e}", exc_info=True)
        return False