# PerfectOCR/core/workflow/ocr/paddle_wrapper.py
import logging
import time
# import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from core.domain.data_models import Polygons
from core.domain.data_formatter import DataFormatter
from core.factory.abstract_worker import OCRAbstractWorker
from core.domain.models_manager import ModelsManager
from core.utils.text_utils import space_removal, validate_text
from core.utils.image_utils import elevate_dims
from services.output_service import save_raw_json

logger = logging.getLogger(__name__)

class PaddleOCRWrapper(OCRAbstractWorker):
    """
    Una instancia de PaddleOCR especializada únicamente en el RECONOCIMIENTO
    de texto en imágenes pre-recortadas (polígonos).
    Utiliza carga perezosa para el motor de PaddleOCR.
    """
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        worker_config = config.get("paddle_wrapper", {})
        self.min_confidence = worker_config.get("min_confidence")
        self.output = config.get("ocr_raw", False)
        self._engine = None
        
    @property
    def engine(self) -> Optional[Any]:
        if self._engine is None:
            paddle_manager = ModelsManager.get_instance()
            self._engine = paddle_manager.recognition_engine
            
            if self._engine is None:
                logger.error("PaddleOCRWrapper: Motor de reconocimiento no disponible en PaddleManager")
            else:
                logger.debug("PaddleOCRWrapper: Motor de reconocimiento obtenido del PaddleManager")
        
        return self._engine
        
    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        start_time = time.perf_counter()
        try:
            polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
            logger.debug(f"[PaddleWrapper] Polígonos obtenidos: {len(polygons)}")

            final_results = self.recognize_text_from_batch(polygons, manager)

            processed_count = 0
            if final_results:
                success = manager.update_ocr_results(final_results)
                processed_count = len(final_results) if success else 0
                
                if self.output:
                    file_name: str = manager.workflow.metadata.image_name #type: ignore
                    worker_name = context.get("worker_name") or "paddle_wrapper"
                    output_paths = context["output_paths"]
                    save_raw_json( output_paths, worker_name, final_results, file_name)
            
            logger.debug(f"Batch OCR completado. {processed_count} polígonos procesados en {time.perf_counter() - start_time:.6f}s.")
            
            return True
        except Exception as e:
            logger.error(f"Error en paddle OCR: {e}", exc_info=True)
        return False
        
    def recognize_text_from_batch(self, polygons: Dict[str, Polygons], manager: DataFormatter) -> Dict[str, Dict[str, Any]]:
        """
        Ejecuta OCR y filtra inmediatamente por confianza para reducir overhead.
        """
        time0 = time.perf_counter()
        if self.engine is None:
            return {}
        try:
            if not polygons:
                return {}
            # polygons_list = [polygon for polygon in polygons.values() if polygon.cropped_img.cropped_img is not None]

            polygon_ids = [pdx.polygon_id for pdx in polygons.values() if pdx.cropped_img.cropped_img is not None]
            img_list = [p.cropped_img.cropped_img for p in polygons.values() if p.cropped_img.cropped_img is not None]
            logger.info(f"Coherencia: {len(polygon_ids) == len(img_list)}")
            image_list = elevate_dims(img_list)
            
            batch_result = self.engine.ocr(image_list, cls=False, det=False, rec=True)[0]
            manager.delete_cropped_images()

            # logger.info(f"BATCH RESULT OCR TYPO:{type(batch_result)}"
            #             "\n"f"Resultado 0: {batch_result}")
            
            raw_map: Dict[str, Dict[str, Any]] = {}
            # confidence_list: List[Tuple[str, float]] = []
            for idx, (text, _) in enumerate(batch_result):
                # confidence_list.append((text, confid))
                clean_text = space_removal(text)
                if validate_text(clean_text):
                    raw_map[polygon_ids[idx]] = {
                        "text": clean_text,
                    }
                else:
                    logger.debug(f"Texto filtrado: '{text}'")
                    continue

            # Ordenar por confianza de mayor a menor
            # confidence_list_sorted = sorted(confidence_list, key=lambda x: x[1], reverse=False)
            # logger.info("Confianzas ordenadas (mayor a menor): "
            #             "\n"f"{confidence_list_sorted[:7]}")

            logger.info(f"Tiempo en OCR: {time.perf_counter() - time0:.6f}'s")
            return raw_map
            
        except TypeError as e:
            logger.error(f"Error en recognize_text_from_batch: {e}", exc_info=True)
        return {}
