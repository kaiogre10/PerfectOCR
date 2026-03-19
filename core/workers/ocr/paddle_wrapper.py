# PerfectOCR/core/workflow/ocr/paddle_wrapper.py
import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional
from core.domain.data_models import Polygons
from core.domain.data_formatter import DataFormatter
from core.factory.abstract_worker import OCRAbstractWorker
from core.domain.models_manager import ModelsManager
from core.utils.text_utils import space_removal, validate_alone_chars, detect_special_strings, separate_punt
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

            image_list: List[np.ndarray[Any, np.dtype[np.uint8]]] = []
            polygon_ids: List[str] = []
            
            for poly_id, polygon in polygons.items():
                cropped_img = polygon.cropped_img.cropped_img if polygon.cropped_img else None
                
                if cropped_img is not None:
                    cropped_img = elevate_dims(cropped_img)
                    image_list.append(cropped_img)
                    polygon_ids.append(poly_id)
            
            if not image_list:
                logger.warning(" No se encontraron imágenes válidas para OCR.")
                return False
                
            raw_results = self.recognize_text_from_batch(image_list, polygon_ids)
            if not manager.delete_cropped_images():
                logger.warning("Cropped images no se liberaron")

            # logger.info("Cropped images liberadas con éxito")

            final_results = self._is_valid_polygon(raw_results)

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
        
    def recognize_text_from_batch(self, image_list: List[np.ndarray[Any, np.dtype[np.uint8]]], polygon_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Ejecuta OCR y filtra inmediatamente por confianza para reducir overhead.
        """
        if self.engine is None or not image_list:
            return {}

        try:
            batch_result = self.engine.ocr(image_list, cls=False, det=False, rec=True)
                                    
            if len(batch_result) == 1 and isinstance(batch_result[0], list):
                consolidated = batch_result[0]
                
                if len(consolidated) == len(image_list):
                    raw_map: Dict[str, Dict[str, Any]] = {}
                    for idx, (text, confidence) in enumerate(consolidated):
                        conf_pct = round(float(confidence) * 100.0, 2)
                        
                        # Filtro de confianza inmediato
                        if conf_pct >= self.min_confidence:
                            raw_map[polygon_ids[idx]] = {
                                "text": text
                            }
                        else:
                            logger.debug(f"Baja confianza en {polygon_ids[idx]}: '{text}' ({conf_pct}%)")
                        
                    return raw_map
            
        except Exception as e:
            logger.error(f"Error en recognize_text_from_batch: {e}")
        return {}

    def _is_valid_polygon(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Lógica de limpieza textual post-filtro de confianza."""
        final_results: Dict[str, Dict[str, Any]] = {}
        for poly_id, data in results.items():
            text = data["text"]
            
            text = space_removal(text)
            clean_text = separate_punt(text)
            
            # Filtro por contenido (Ya no repetimos el de confianza)
            if clean_text and not detect_special_strings(clean_text) and validate_alone_chars(clean_text):
                final_results[poly_id] = {"text": clean_text}
        
        return final_results
