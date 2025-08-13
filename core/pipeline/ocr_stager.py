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
from core.domain.data_models import Polygons


logger = logging.getLogger(__name__)

class OCRStager:
    def __init__(self, stage_config: Dict[str, Any], paddleocr: PaddleOCRWrapper, project_root: str):
        self.stage_config = stage_config
        self.project_root = project_root
        self.paddleocr = paddleocr
        self.paddle = paddleocr
        
    def run_ocr_on_polygons( 
        self, manager: DataFormatter
    ) -> Tuple[Optional[DataFormatter], float]:
        
        start_time = time.perf_counter()
        
        cropped_images = manager.get_cropped_images_for_preprocessing()
        
        
        if not self.paddle or self.paddle.engine is None:
            return manager, 0.0

        if not manager.get_cropped_images_for_preprocessing():
            logger.warning("[OCREngineManager] No polygons were provided to OCREngineCoordinator.")
            return manager, 0.0

        image_list = []
        polygon_ids = []
        for poly_id, polygon in cropped_images.items():
            
                # Convertir imagen a RGB si está en escala de grises
            img = polygon.cropped_img
            if img is None:
                logger.warning(f"La imagen recortada para el polígono {poly_id} es nula, se omitirá.")
                continue

            if not isinstance(img, np.ndarray):
                img = np.array(img)

            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            image_list.append(img)
            polygon_ids.append(poly_id)

        if not image_list:
            logger.warning("[OCREngineManager] No se encontraron polígonos con imágenes válidas para procesar.")
            return manager, 0.0

        batch_results = self.paddle.recognize_text_from_batch(image_list)
        
        processed_count = 0
        if batch_results:
            for i, result in enumerate(batch_results):
                if i < len(polygon_ids):
                    poly_id = polygon_ids[i]
                    if result and manager.workflow and poly_id in manager.workflow.image_data.polygons:
                        polygon = manager.workflow.image_data.polygons[poly_id]
                        polygon.ocr_text = result.get("text")
                        polygon.ocr_confidence = result.get("confidence")
                        processed_count += 1
        
        logger.info(f"[OCREngineManager] Lote procesado. Resultados para {processed_count}/{len(image_list)} polígonos.")
        
        ocr_time = time.perf_counter() - start_time
        
        
        return manager, ocr_time


    def _save_complete_ocr_results(self, manager: DataFormatter, image_name: str):
        """
        Guarda los resultados OCR en formato JSON, solo con polygon_id, texto, confianza y metadata básica.
        """
        if not self.stage_config.get('output_flag', {}).get('ocr_raw', False):
            return

        output_folder = self.stage_config.get('output_folder')
        if not output_folder:
            logger.warning("[OCREngineManager] No se puede guardar resultados OCR porque 'output_folder' no está definido.")
            return

        try:
            os.makedirs(output_folder, exist_ok=True)
            base_name = image_name
            json_filename = f"{base_name}_ocr_results.json"
            json_path = os.path.join(output_folder, json_filename)

            if not manager.workflow:
                logger.warning("No se puede guardar resultados OCR porque el workflow en DataFormatter no está inicializado.")
                return

            # Metadata básica
            metadata = manager.workflow.metadata
            output_data = {
                "doc_name": metadata.image_name if metadata else None,
                "formato": metadata.format if metadata else None,
                "dpi": metadata.dpi if metadata else None,
                "img_dims": {
                    "width": metadata.img_dims.get("width") if metadata and metadata.img_dims else None,
                    "height": metadata.img_dims.get("height") if metadata and metadata.img_dims else None
                } if metadata and metadata.img_dims else None,
                "fecha_creacion": str(metadata.date_creation) if metadata else None,
                "polygons": []
            }

            # Solo polygon_id, texto y confianza
            polygons_data = manager.workflow.image_data.polygons
            for pid, p in polygons_data.items():
                output_data["polygons"].append({
                    "polygon_id": pid,
                    "text": p.ocr_text,
                    "confidence": p.ocr_confidence
                })

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            logger.info(f"[OCREngineManager] Resultados OCR guardados en {json_path}")
        except Exception as e:
            logger.error(f"[OCREngineManager] Error guardando resultados OCR: {e}")