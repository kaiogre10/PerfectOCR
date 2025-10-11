# PerfectOCR/core/workflow/ocr/paddle_wrapper.py
import cv2
import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional
from core.domain.data_models import Polygons
from core.domain.data_formatter import DataFormatter
from core.factory.abstract_worker import OCRAbstractWorker
from core.domain.models_manager import ModelsManager

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
        self.config = config
        self.worker_config = config.get("paddle_wrapper", {})
        self.enabled_outputs = config.get("enabled_outputs", {})
        self.output = self.enabled_outputs.get("ocr_raw", False)
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
        polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
        logger.debug(f"[PaddleWrapper] Polígonos obtenidos: {len(polygons)}")
        image_list: List[np.ndarray[Any, np.dtype[np.uint8]]] = []
        polygon_ids: List[str] = []
        
        for poly_id, polygon in polygons.items():
            cropped_img = polygon.cropped_img.cropped_img if polygon.cropped_img else None
            
            if cropped_img is not None:
                if len(cropped_img.shape) == 2:
                    cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2BGR)
                elif cropped_img.shape[2] == 1:
                    cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2BGR)
                
                image_list.append(cropped_img)
                polygon_ids.append(poly_id)
        
        if not image_list:
            logger.warning("[PaddleWrapper] No se encontraron imágenes válidas para OCR.")
            return False
            
        final_results: List[Optional[Dict[str, Any]]] = self.recognize_text_from_batch(image_list)
        processed_count = 0
        
        if final_results:
            success = manager.update_ocr_results(final_results, polygon_ids)
            processed_count = len(final_results) if success else 0
            manager.clear_cropped_images(polygon_ids)
            logger.debug("Cropped_img liberadas usando DataFormatter")
            logger.debug(f"{final_results}")
            
            if self.output:
                file_name: str = manager.workflow.metadata.image_name
                self._save_ocr_raw(context, final_results, file_name)
        
        total_time = time.perf_counter() - start_time
        logger.debug(f"Batch OCR completado. {processed_count}/{len(image_list)} polígonos procesados en {total_time:.4f}s.")
        return True
        
    def recognize_text_from_batch(self, image_list: List[np.ndarray[Any, np.dtype[np.uint8]]]) -> List[Optional[Dict[str, Any]]]:
        """
        Ejecuta OCR en un lote (batch) de imágenes pre-recortadas.
        Está adaptado para manejar el caso en que PaddleOCR devuelve una única
        lista consolidada de resultados.
        """
        min_confidence = self.worker_config.get("min_confidence")
        if self.engine is None:
            logger.error("PaddleOCR recognition engine not initialized. Cannot recognize text.")
            return [None] * len(image_list)
        
        if not image_list:
            logger.warning("Se recibió una lista vacía de imágenes para el reconocimiento por lotes.")
            return []

        try:
            valid_images: List[np.ndarray[Any, np.dtype[np.uint8]]] = []
            for idx, img in enumerate(image_list):
                if img is None or not hasattr(img, "shape") or len(img.shape) < 2 or img.size == 0:
                    logger.warning(f"Imagen inválida en el batch (índice {idx}): {type(img)} - shape: {getattr(img, 'shape', None)}")
                    continue
                valid_images.append(img)
            if not valid_images:
                logger.error("No hay imágenes válidas para el reconocimiento por lotes.")
                return []
            
            batch_result: List[List[str]] = self.engine.ocr(valid_images, cls=False, det=False, rec=True) 
                                    
            if len(batch_result) == 1 and isinstance(batch_result[0], list):
                consolidated_results = batch_result[0]
                
                if len(consolidated_results) == len(valid_images):
                    logger.debug(f"Resultado consolidado detectado. Mapeando {len(consolidated_results)} textos a {len(valid_images)} imágenes por orden.")
                    final_results: List[Optional[Dict[str, Any]]] = []
                    
                    for text, confidence in consolidated_results:
                        confidence_pct = round(float(confidence) * 100.0, 2) if isinstance(confidence, (float, int)) else 0.0
                        
                        # Aplicar filtro de confianza mínima
                        if confidence_pct < min_confidence:
                            logger.debug(f"Resultado filtrado por baja confianza:'{text}'-> {confidence_pct}% < {min_confidence}%")
                            final_results.append(None)
                            continue
                            
                        processed_result: Dict[str, Any] = {
                            "text": str(text).strip(),
                            "confidence": confidence_pct
                        }
                        final_results.append(processed_result)
                    
                    logger.debug(f"Total de resultados finales procesados: {len(final_results)}")
                    return final_results
                else:
                    logger.error(f"Error de mapeo: El lote devolvió {len(consolidated_results)} textos para {len(image_list)} imágenes. No se puede garantizar la correspondencia.")
                    return [None] * len(image_list)
            
        except Exception as e:
            logger.error(f"Error crítico durante el reconocimiento de texto en lote: {e}", exc_info=True)
            return [None] * len(image_list)

    def _save_ocr_raw(self, context: Dict[str, Any], final_results: List[Optional[Dict[str, Any]]], file_name: str):
        from services.output_service import save_json
        import os
        project_root = self.project_root
        output_file = context.get("output_paths", [])
        for path in output_file:
            output_dir: str = os.path.join(path, "ocr_raw")
            json_file_name = f"{os.path.splitext(file_name)[0]}.json"
            output_file = save_json(final_results, output_dir, json_file_name, project_root)
        if output_file:
            logger.debug(f"OCR Raw results para '{file_name}' guardado en {len(output_file)} ubicaciones.")
