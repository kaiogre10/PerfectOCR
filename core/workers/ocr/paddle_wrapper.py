# PerfectOCR/core/workflow/ocr/paddle_wrapper.py
import logging
import time
from typing import Dict, Any, Optional, List
from core.domain.data_models import Polygons
from core.domain.data_formatter import DataFormatter
from core.factory.abstract_worker import OCRAbstractWorker
from app.models_builder import ModelsBuilder
from core.utils.text_utils import normalice_text
from core.utils.image_utils import elevate_dims, make_contiguous
from services.output_service import save_text_debug
from core.utils.compiled_utils import validate_text

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
        self.output = config.get("ocr_raw")
        self.del_output_log = config.get("text_del")
        self._engine = None
        
    @property
    def engine(self) -> Optional[Any]:
        if self._engine is None:
            paddle_manager = ModelsBuilder.get_instance()
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
                worker_name = context.get("worker_name") or "paddle_wrapper"
                success = manager.update_ocr_results(final_results, worker_name)
                processed_count = len(final_results) if success else 0
                
                if self.output:
                    file_name: str = manager.workflow.metadata.image_name if manager.workflow else ""
                    file_name = f"{file_name}"
                    save_text_debug(worker_name, final_results, file_name)
            
            logger.debug(f"Batch OCR completado. {processed_count} polígonos procesados en {time.perf_counter() - start_time:.6f}s.")
            
            return True
        except Exception as e:
            logger.error(f"Error en paddle OCR: {e}", exc_info=True)
            context = {}
        return False
        
    def recognize_text_from_batch(self, polygons: Dict[str, Polygons], manager: DataFormatter) -> Dict[str, Dict[str, Any]]:
        """Ejecuta OCR y filtra inmediatamente por confianza para reducir overhead."""
        if self.engine is None:
            return {}
        time0 = time.perf_counter()
        try:
            polygon_ids = [pdx.polygon_id for pdx in polygons.values() if pdx.cropped_img.cropped_img is not None]
            img_list = [make_contiguous(p.cropped_img.cropped_img) for p in polygons.values() if p.cropped_img.cropped_img is not None]
            image_list = elevate_dims(img_list)
            manager.delete_cropped_images()
            
            time_t = time.perf_counter()
            batch_result = self.engine.ocr(image_list, cls=False, det=False, rec=True)
            logger.info(f"Transcripción completa en: '{time.perf_counter() - time_t}'s'")
            image_list = None
            deleted: List[List[str]] = []
            raw_map: Dict[str, Dict[str, Any]] = {}

            idx = 0
            for word_tuple in batch_result:
                logger.info(f"WORDS: '{word_tuple}'")
                text = word_tuple[0]
                confidence = word_tuple[1]
                idx += 1
                if not text or not validate_text(text):
                    deleted.append([polygon_ids[idx], text])
                    # logger.info(f"INVÁLIDO: {polygon_ids[idx]} '{text}'")
                    continue
                
                elif confidence < self.min_confidence:
                    deleted.append([polygon_ids[idx], text])
                    if self.del_output_log:
                        logger.info(f"BAJA CONFIANZA: {polygon_ids[idx]} | '{text}' | '{(confidence*100.0):.4f}%' ")
                    continue
                else:
                    norm_text = normalice_text(text, False)
                    raw_map[polygon_ids[idx]] = {"text": norm_text.strip()}
                    # logger.info(f"OCR FILTRO: {polygon_ids[idx]}: '{text}' -> '{norm_text}', CONF: {confidence*100.0} %")
                    continue

            # for id, data in raw_map.items():
            #     logger.info(f"OCR FILTRO: {id} {data.get("text", "")}")
                
            # logger.debug(f"PADDLE OCR COMPLETO EN: {time.perf_counter() - time0:.6f}'s")
            return raw_map
            
        except TypeError as e:
            logger.error(f"Error en recognize_text_from_batch: {e}", exc_info=True)
        return {}
