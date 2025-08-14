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
    def __init__(self, stage_config: Dict[str, Any], paddleocr: PaddleOCRWrapper, project_root: str):
        self.stage_config = stage_config
        self.project_root = project_root
        self.paddleocr = paddleocr
        self.paddle = paddleocr
        
    def run_ocr_on_polygons( 
        self, manager: DataFormatter
    ) -> Tuple[Optional[DataFormatter], float]:
        start_time = time.perf_counter()

        if not self.paddle or self.paddle.engine is None:
            logger.error("[OCREngineManager] Engine PaddleOCR no inicializado.")
            return manager, 0.0

        cropped_images = manager.get_cropped_images_for_preprocessing()
        if not isinstance(cropped_images, dict) or not cropped_images:
            logger.warning("[OCREngineManager] No hay imágenes recortadas para OCR.")
            return manager, 0.0

        image_list: List[Any] = []  # type: ignore
        polygon_ids: List[str] = []
        for poly_id, cropped in cropped_images.items():
            try:
                img = getattr(cropped, "cropped_img", None)
                if img is None:
                    continue
                # Validar tamaño mínimo
                if hasattr(img, 'shape') and isinstance(img.shape, tuple) and len(img.shape) >= 2:
                    if min(img.shape[:2]) == 0:
                        continue
                image_list.append(img)
                polygon_ids.append(poly_id)
            except Exception as e:
                logger.error(f"Error preparando imagen de polígono {poly_id} para OCR: {e}")
                continue

        if not image_list:
            logger.warning("[OCREngineManager] No se encontraron imágenes válidas para OCR.")
            return manager, 0.0

        batch_results = self.paddle.recognize_text_from_batch(image_list)

        processed_count = 0
        if batch_results:
            for idx, res in enumerate(batch_results):
                if idx >= len(polygon_ids):
                    break
                poly_id = polygon_ids[idx]
                if not res or not manager.workflow_dict or poly_id not in manager.workflow_dict:
                    continue
                try:
                    # Estructura esperada: {"text": str, "confidence": float}
                    manager.workflow_dict[poly_id]["ocr_text"] = res.get("text", "")
                    manager.workflow_dict[poly_id]["ocr_confidence"] = res.get("confidence")
                    processed_count += 1
                except Exception as e:
                    logger.error(f"Error asignando resultado OCR a polígono {poly_id}: {e}")

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
            polygons_data = manager.workflow_dict.polygons
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