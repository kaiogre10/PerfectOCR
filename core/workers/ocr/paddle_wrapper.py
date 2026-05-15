# PerfectOCR/core/workflow/ocr/paddle_wrapper.py
import logging
import time
from typing import Dict, Any, Optional
from core.domain.data_models import Polygons
from core.domain.data_formatter import DataFormatter
from core.factory.abstract_worker import OCRAbstractWorker
from core.domain.models_manager import ModelsManager
from core.utils.text_utils import validate_text, normalice_text
from core.utils.image_utils import elevate_dims, make_contiguous
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
                    save_raw_json(worker_name, final_results, file_name)
            
            logger.debug(f"Batch OCR completado. {processed_count} polígonos procesados en {time.perf_counter() - start_time:.6f}s.")
            
            return True
        except Exception as e:
            logger.error(f"Error en paddle OCR: {e}", exc_info=True)
        return False
        
    def recognize_text_from_batch(self, polygons: Dict[str, Polygons], manager: DataFormatter) -> Dict[str, Dict[str, Any]]:
        """
        Ejecuta OCR y filtra inmediatamente por confianza para reducir overhead.
        """
        if self.engine is None:
            return {}
        time0 = time.perf_counter()
        try:
            polygon_ids = [pdx.polygon_id for pdx in polygons.values() if pdx.cropped_img.cropped_img is not None]
            img_list = [make_contiguous(p.cropped_img.cropped_img) for p in polygons.values() if p.cropped_img.cropped_img is not None]
            image_list = elevate_dims(img_list)
            manager.delete_cropped_images()
            
            # time_t = time.perf_counter()
            batch_result = self.engine.ocr(image_list, cls=False, det=False, rec=True)
            # logger.info(f"Transcripción completa en: '{time.perf_counter() - time_t}'s'")
            image_list = None
            # deleted: List[List[str]] = []
            raw_map: Dict[str, Dict[str, Any]] = {}

            for idx, (text, confidence) in enumerate(batch_result[0]):
                if not text or not validate_text(text):
                    # deleted.append([polygon_ids[idx], text])
                    # logger.info(f"INVÁLIDO: {polygon_ids[idx]} '{text}'")
                    continue
                
                elif confidence < self.min_confidence:
                    # deleted.append([polygon_ids[idx], text])
                    # logger.info(f"BAJA CONFIANZA: {polygon_ids[idx]} {confidence*100.0} % | '{text}'")
                    continue
                else:
                    text = normalice_text(text)
                    raw_map[polygon_ids[idx]] = {"text": text}
            # logger.debug(f"PADDLE OCR COMPLETO EN: {time.perf_counter() - time0:.6f}'s")
            return raw_map
            
        except TypeError as e:
            logger.error(f"Error en recognize_text_from_batch: {e}", exc_info=True)
        return {}
