# PerfectOCR/core/coordinators/ocr_manager.py
import os
import numpy as np
import logging
import time
import json
from core.domain.data_formatter import DataFormatter
from typing import Optional, Dict, Any, Tuple, List
from core.workers.ocr.paddle_wrapper import PaddleOCRWrapper
import cv2

logger = logging.getLogger(__name__)

class OCRStager:
    def __init__(self, stage_config: Dict[str, Any], paddleocr: PaddleOCRWrapper, output_paths: Optional[List[str]], project_root: str):
        self.stage_config = stage_config
        self.project_root = project_root
        self.paddleocr = paddleocr
        self.paddle = paddleocr
        self.output_paths = output_paths if output_paths is not None else []
        
    def run_ocr_on_polygons(self, manager: DataFormatter) -> Tuple[Optional[DataFormatter], float]:
        start_time = time.perf_counter()
        
        polygons = manager.get_polygons_with_cropped_img()
        
        # DEBUG: Ver qué contiene polygons
        logger.info(f"[OCREngineManager] Polígonos obtenidos: {len(polygons)}")
        for poly_id, poly_data in list(polygons.items())[:3]:  # Solo los primeros 3
            cropped_img = poly_data.get("cropped_img", {})
            logger.debug(f"[OCREngineManager] {poly_id}: cropped_img type={type(cropped_img)}, shape={getattr(cropped_img, 'shape', 'N/A')}")
        
        # Preparar batch (igual que preprocessing pero optimizado para OCR)
        image_list: List[np.ndarray[Any, Any]] = []
        polygon_ids: List[str] = []
        
        for poly_id, poly_data in polygons.items():
            cropped_img = poly_data.get("cropped_img", {})
            cropped_img: np.ndarray[Any, Any]
            if cropped_img is not None:
                # Convertir a np.ndarray si es necesario
                if isinstance(cropped_img, list):
                    cropped_img = np.array(cropped_img)
                
                # Validar que la imagen sea procesable
                if hasattr(cropped_img, 'shape') and len(cropped_img.shape) >= 2:
                    if min(cropped_img.shape[:2]) > 0:  # Dimensiones válidas
                        # CONVERTIR A 3 CANALES para PaddleOCR
                        if len(cropped_img.shape) == 2:  # Escala de grises
                            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2BGR)
                        elif cropped_img.shape[2] == 1:  # 1 canal
                            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2BGR)
                        
                        image_list.append(cropped_img)
                        polygon_ids.append(poly_id)
        
        if not image_list:
            logger.warning("[OCREngineManager] No se encontraron imágenes válidas para OCR.")
            return manager, 0.0
        
        # Procesar BATCH (mantener rendimiento)
        batch_results: List[Optional[Dict[str, Any]]] = self.paddle.recognize_text_from_batch(image_list)
        
        # Actualizar resultados usando el método centralizado
        processed_count = 0
        if batch_results:
            success = manager.update_ocr_results(batch_results, polygon_ids)
            processed_count = len(batch_results) if success else 0

            for poly_id in polygon_ids:
                if poly_id in manager.get_polygons():
                    manager.workflow_dict["polygons"][poly_id]["cropped_img"] = None
                    #logger.debug("Cropped_img liberadas, texto generado")
        
        logger.info(f"[OCREngineManager] Batch OCR completado. {processed_count}/{len(image_list)} polígonos procesados.")
        image_name = manager.get_metadata().get("image_name", "unknown_image")
        self._save_complete_ocr_results(manager, image_name)
        
        ocr_time = time.perf_counter() - start_time
        return manager, ocr_time


    def _save_complete_ocr_results(self, manager: DataFormatter, image_name: str):
        """
        Ordena al OutputService que guarde los resultados del OCR.
        """
        if not self.stage_config.get('output_flag', {}).get('ocr_raw', False):
            return

        output_folder = self.stage_config.get('output_folder')
        if not output_folder:
            logger.warning("[OCREngineManager] No se puede guardar resultados OCR porque 'output_folder' no está definido.")
            return

        try:
            from services.output_service import save_ocr_json
            save_ocr_json(manager, output_folder, image_name)
        except Exception as e:
            logger.error(f"[OCREngineManager] Fallo al invocar save_ocr_json: {e}")