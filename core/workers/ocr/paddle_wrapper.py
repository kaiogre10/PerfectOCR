# PerfectOCR/core/workflow/ocr/paddle_wrapper.py
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
        try:
            polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
            logger.debug(f"[PaddleWrapper] Polígonos obtenidos: {len(polygons)}")

            image_list: List[np.ndarray[Any, np.dtype[np.uint8]]] = []
            polygon_ids: List[str] = []
            
            for poly_id, polygon in polygons.items():
                cropped_img = polygon.cropped_img.cropped_img if polygon.cropped_img else None
                
                if cropped_img is not None:
                    if len(cropped_img.shape) == 2:
                        import cv2
                        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2BGR)
                    elif cropped_img.shape[2] == 1:
                        import cv2
                        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2BGR)
                    
                    image_list.append(cropped_img) #type: ignore
                    polygon_ids.append(poly_id)
            
            if not image_list:
                logger.warning(" No se encontraron imágenes válidas para OCR.")
                return False
                
            final_results: Dict[str, Dict[str, Any]] = self.recognize_text_from_batch(image_list, polygon_ids)
            processed_count = 0
            
            if final_results:
                success = manager.update_ocr_results(final_results)
                processed_count = len(final_results) if success else 0
                
                if self.output:
                    from services.output_service import save_raw_json
                    file_name: str = manager.workflow.metadata.image_name #type: ignore
                    worker_name = context.get("worker_name") or "paddle_wrapper"
                    output_paths = context["output_paths"]
                    save_raw_json( output_paths, worker_name, final_results, file_name)
            
            total_time = time.perf_counter() - start_time
            logger.debug(f"Batch OCR completado. {processed_count}/{len(image_list)} polígonos procesados en {total_time:.6f}s.")
            
            return True
        except Exception as e:
            logger.error(f"Error en paddle OCR: {e}", exc_info=True)
        return False
        
    def recognize_text_from_batch(self, image_list: List[np.ndarray[Any, np.dtype[np.uint8]]], polygon_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Ejecuta OCR en un lote (batch) de imágenes pre-recortadas.
        Está adaptado para manejar el caso en que PaddleOCR devuelve una única
        lista consolidada de resultados.
        """
        min_confidence = self.worker_config.get("min_confidence")
        if self.engine is None:
            logger.error("PaddleOCR recognition engine not initialized. Cannot recognize text.")
            return {}
        
        if not image_list:
            logger.warning("Se recibió una lista vacía de imágenes para el reconocimiento por lotes.")
            return {}

        try:
            valid_images: List[np.ndarray[Any, np.dtype[np.uint8]]] = []
            for idx, img in enumerate(image_list):
                if img is None or not hasattr(img, "shape") or len(img.shape) < 2 or img.size == 0: #type: ignore
                    logger.warning(f"Imagen inválida en el batch (índice {idx}): {type(img)} - shape: {getattr(img, 'shape', None)}")
                    return {}
                    
                valid_images.append(img)
                
            if not valid_images:
                logger.error("No hay imágenes válidas para el reconocimiento por lotes.")
                return {}
            
            batch_result: List[List[str]] = self.engine.ocr(valid_images, cls=False, det=False, rec=True)
                                    
            if len(batch_result) == 1 and isinstance(batch_result[0], list): #type: ignore
                consolidated_results = batch_result[0]
                
                if len(consolidated_results) == len(valid_images):
                    final_results: Dict[str, Dict[str, Any]] = {}
                    
                    for idx, (text, confidence) in enumerate(consolidated_results):
                        poly_id = polygon_ids[idx]
                        confidence_pct = round(float(confidence) * 100.0, 2) if isinstance(confidence, (float, int)) else 0.0
                        
                        # Aplicar filtro de confianza mínima
                        if confidence_pct > min_confidence:
                            final_results[poly_id] = {
                                "text": str(text).strip(),
                                "confidence": confidence_pct
                            }
                            logger.debug(f"Resultados: {poly_id}: Texto='{text}', Confianza='{confidence_pct}%'")

                        else:
                            logger.debug(f"Texto basuta filtrado en {poly_id}: '{text}' -> '{confidence_pct}%' < '{min_confidence}%'")
                        
                    total_results = len(final_results)
                    logger.warning(f"Se mapearon: '{total_results}' y se descartaron: '{len(consolidated_results) - total_results}' polígonos")
                    return final_results
                else:
                    logger.error(f"Error de mapeo: El lote devolvió {len(consolidated_results)} textos para {len(image_list)} imágenes.")
                    return {}
            
        except Exception as e:
            logger.error(f"Error crítico durante el reconocimiento de texto en lote: {e}", exc_info=True)
        return {}