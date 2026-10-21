# PerfectOCR/core/workflow/ocr/paddle_wrapper.py
import logging
from typing import Dict, Any, Optional, List, Tuple
from domain.data_formatter import DataFormatter
from core.contracts.abstract_worker import OCRAbstractWorker
from app.models_builder import ModelsBuilder
from utils.text_utils import normalice_text
from utils.image_utils import elevate_dims
from services.output_service import save_text_debug
from utils.compiled_utils import validate_text, space_removal

logger = logging.getLogger(__name__)

class PaddleOCRWrapper(OCRAbstractWorker):
    """
    Una instancia de PaddleOCR especializada únicamente en el RECONOCIMIENTO de texto en imágenes pre-recortadas (polígonos).
    Utiliza carga perezosa para el motor de PaddleOCR.
    """
    __slots__ = (
        # "project_root",
        "min_confidence",
        "output",
        "del_output_log",
        "_engine"
    )
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        worker_config = config.get("paddle_wrapper", {})
        self.min_confidence = worker_config.get("min_confidence")
        self.output = config.get("ocr_raw")
        self.del_output_log = config.get("text_del")
        self.text_ocr_log = config.get("text_ocr")
        self._engine = None
        
    @property
    def engine(self) -> Optional[Any]:
        if self._engine is None:
            paddle_manager = ModelsBuilder.get_instance()
            self._engine = paddle_manager.recognition_engine
            
            if self._engine is None:
                raise ModuleNotFoundError("Motor de reconocimiento no disponible en PaddleManager")
            else:
                logger.debug("Motor de reconocimiento obtenido del PaddleManager")
        
        return self._engine
        
    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        try:
            final_results = self.recognize_text_from_batch(manager)

            if final_results:
                if manager.update_ocr_results(final_results):
                    logger.debug(f"Batch OCR completado. {len(final_results)} polígonos")
                    if self.output:
                        file_name = manager.workflow.image_name if manager.workflow else ""
                        save_text_debug(final_results, file_name)
                    return True
                
            return False
        except Exception as e:
            logger.error(f"Error en paddle OCR: {e}", exc_info=True)
            return False
        
    def recognize_text_from_batch(self, manager: DataFormatter) -> Dict[str, Dict[str, Any]]:
        """Ejecuta OCR y filtra inmediatamente por confianza para reducir overhead."""
        try:
            if self.engine is None:
                raise ImportError("PADDLE REC NO DISPONIBLE")
            
            polygons = manager.workflow.polygons if manager.workflow else {}
            if not polygons:
                logger.error("No hay polygons para procesar", exc_info=True)
                raise
            
            logger.debug(f"[PaddleWrapper] Polígonos obtenidos: {len(polygons)}")
            polygon_ids = [pdx.polygon_id for pdx in polygons.values() if pdx.cropped_img is not None]
            img_list = [p.cropped_img for p in polygons.values() if p.polygon_id in polygon_ids]
            image_list = elevate_dims(img_list)
            manager.delete_cropped_images()
            if not image_list:
                logger.error(f"NO SE ELEVARON CORRECTAMENTE LAS DIMENSIONES")
                return {}
            
            # time_t = time.perf_counter()
            batch_result: List[Tuple[str, float]] = self.engine.ocr(image_list, cls=False, det=False, rec=True)[0]
            image_list = None
            del image_list
            # logger.info(f"Transcripción completa en: '{time.perf_counter() - time_t}'s'")
            if not batch_result:
                logger.error(f"PADDLE NO ARROJO RESULTADOS")
                return {}
            
            deleted: List[List[str]] = []
            raw_map: Dict[str, Dict[str, Any]] = {}

            for idx, (text, confidence) in enumerate(batch_result):
                poly_id = polygon_ids[idx]
                if self.text_ocr_log:
                    logger.info(f"TEXTO CRUDO: {poly_id} | '{text}' | CONF: {confidence*100.0}:.6f%")

                if not text:
                    deleted.append([poly_id, text])
                    if self.del_output_log:
                        logger.info(f"INVÁLIDO: {poly_id} '{text}'")
                    continue
                
                elif confidence < self.min_confidence:
                    deleted.append([poly_id, text])
                    if self.del_output_log:
                        logger.info(f"BAJA CONFIANZA: {poly_id} | '{text}' | '{(confidence*100.0)}:.6f%' ")
                    continue
                
                else:
                    if text.isascii():
                        if not " " in text:
                            norm_text = text
                        else:
                            norm_text = space_removal(text)
                    else:
                        norm_text = normalice_text(text)
                        
                    if not norm_text or not validate_text(norm_text):
                        if self.del_output_log:
                            logger.info(f"TEXTO INVALIDO: {poly_id}: '{text}' -> '{norm_text}', CONF: {confidence*100.0}:.6f%")
                        continue

                    raw_map[poly_id] = {"text": norm_text}
                    continue

            return raw_map
            
        except Exception as e:
            logger.error(f"Error en recognize_text_from_batch: {e}", exc_info=True)
        return {}

    # def procesar_pagina_pdf(self, ruta_pdf: str):
    #     # 1. INTENTO DE SUPLANTACIÓN (Costo de CPU/RAM: Casi 0)
    #     with pdfplumber.open(ruta_pdf) as pdf:
    #         texto_vectorial: List[str] = []
    #         for pagina in pdf.pages:
    #             text_lines = pagina.extract_text_lines()
    #             for line in text_lines:
    #                 line_text: str = line.get("text", "")
    #                 if validate_text(line_text):
    #                     texto_vectorial.append(line_text)
    #                     # logger.info(f"{line_text}")
    #
    #     logger.info(f"TEXTO VECTORIAL: '{texto_vectorial}'")
    #     return texto_vectorial
