# core/workers/ocr/text_refiner.py
from typing import Dict, Any, Optional, List, Tuple
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons
from core.factory.abstract_worker import OCRAbstractWorker
from core.workers.ocr.text_cleaner import TextCleaner
from core.workers.ocr.fragmenter import Fragmenter
from core.workers.ocr.text_corrector import TextCorrector
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
    def __init__(self, config: Dict[str, Any], project_root: str, cleaner: Optional[TextCleaner] = None, corrector: Optional[TextCorrector] = None, fragmenter: Optional[Fragmenter] = None):
        super().__init__(config, project_root)
        self.worker_config = config.get("text_refiner", {})
        self.output = config.get("cleanned_text")
        self.seman_clas_log = config.get("seman_clas")
        self.semantic_types_log: List[int] = [-1, 0, 1, 2, 3, 4, 5]
        self.num_passes = self.worker_config.get("num_passes")
        self.cleaner = cleaner
        self.fragmenter = fragmenter
        self.corrector = corrector

    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """
        Ejecuta el ciclo de refinamiento con clasificación selectiva.
        """
        t0 = time.perf_counter()
        self.preprocess_text(manager)
        self.get_early_data(manager)
        
        if self.num_passes == 0:
            self.classify_strings(manager)
        else:            
            for i in range(self.num_passes):
                pass_num = i + 1
                # logger.debug(f"Iniciando Bucle de Refinamiento de Texto #{pass_num}")

                # logger.debug(f"Pasada 1, bucle #{pass_num}: Clasificación Semántica")
                # step_t0 = time.perf_counter()
                # self.classify_strings(manager)
                # self._log_worker_time(pass_num, "SemanticClassifier", step_t0, "Clasificación Semántica")
                
                if self.cleaner:
                    self.cleaner.transcribe(context, manager)
                    # logger.info(f"Pasada 1, bucle #{pass_num}: Clasificación Semántica (solo limpios)")
                    self.classify_strings(manager)

                if self.corrector:
                    self.corrector.transcribe(context, manager)
                    self.classify_strings(manager)
                
                if self.fragmenter:
                    self.fragmenter.transcribe(context, manager)
                    self.classify_strings(manager)

        # logger.info(f"Pasada final: Clasificación Semántica completa")
        logger.info(f"Tiempo de refinado: {time.perf_counter() - t0:.6f}'s")
        if self.seman_clas_log:
            polygons = manager.workflow.polygons if manager.workflow else {}
            for poly, poly_data in polygons.items():
                if any(sc in self.semantic_types_log for sc in  poly_data.semantic_clasification):
                    logger.info(f"{poly}: '{poly_data.ocr_text}', clas: {poly_data.semantic_clasification}")
        
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
            state: List[bool] = [False, False, False, False, False, False]

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
                if 12 in keys:
                    state[5] = True

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