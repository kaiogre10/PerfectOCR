# PerfectOCR/core/coordinators/ocr_manager.py
import os
import numpy as np
import logging
import time
import json
from typing import Optional, Dict, Any, Tuple
from core.workers.ocr.paddle_wrapper import PaddleOCRWrapper
import cv2

logger = logging.getLogger(__name__)

class OCRStager:
    def __init__(self, stage_config: Dict[str, Any], paddleocr: PaddleOCRWrapper, project_root: str):
        self.stage_config = stage_config
        self.project_root = project_root
        self.paddleocr = paddleocr
        self.paddle = paddleocr
        
    def _save_complete_ocr_results(self, workflow_job, image_name: str):
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

            # Metadata básica
            metadata = workflow_job.doc_metadata
            output_data = {
                "doc_name": metadata.doc_name if metadata else None,
                "formato": metadata.formato if metadata else None,
                "dpi": metadata.dpi if metadata else None,
                "img_dims": {
                    "width": metadata.img_dims.width,
                    "height": metadata.img_dims.height
                } if metadata and metadata.img_dims else None,
                "fecha_creacion": str(metadata.date_creation) if metadata else None,
                "polygons": []
            }

            # Solo polygon_id, texto y confianza
            for pid, p in workflow_job.polygons.items():
                output_data["polygons"].append({
                    "polygon_id": pid,
                    "text": getattr(p, "text", None),
                    "confidence": getattr(p, "confidence", None)
                })

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            logger.info(f"[OCREngineManager] Resultados OCR guardados en {json_path}")
        except Exception as e:
            logger.error(f"[OCREngineManager] Error guardando resultados OCR: {e}")

    def run_ocr_on_polygons(
        self, 
        workflow_job: WorkflowJob
    ) -> Tuple[Optional[WorkflowJob], float]:
        
        start_time = time.perf_counter()
        
        if not self.paddle or self.paddle.engine is None:
            workflow_job.add_error("PaddleOCR recognition engine not available.")
            logger.error("[OCREngineManager] PaddleOCR recognition engine not available.")
            return workflow_job, 0.0

        if not workflow_job.polygons:
            logger.warning("[OCREngineManager] No polygons were provided to OCREngineCoordinator.")
            return workflow_job, 0.0

        images_batch = []
        polygon_ids = []
        for poly_id, polygon in workflow_job.polygons.items():
            if polygon.cropped_img is not None:
                # Convertir imagen a RGB si está en escala de grises
                img = polygon.cropped_img
                if len(img.shape) == 2:  # Grayscale
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                images_batch.append(img)
                polygon_ids.append(poly_id)

        if not images_batch:
            logger.warning("[OCREngineManager] No se encontraron polígonos con imágenes válidas para procesar.")
            return workflow_job, 0.0

        batch_results = self.paddle.recognize_text_from_batch(images_batch)
        
        processed_count = 0
        for i, result in enumerate(batch_results):
            poly_id = polygon_ids[i]
            if result and poly_id in workflow_job.polygons:
                workflow_job.polygons[poly_id].text = result.get("text")
                workflow_job.polygons[poly_id].confidence = result.get("confidence")
                processed_count += 1
        
        logger.info(f"[OCREngineManager] Lote procesado. Resultados para {processed_count}/{len(images_batch)} polígonos.")
        
        if workflow_job.doc_metadata:
            self._save_complete_ocr_results(workflow_job, workflow_job.doc_metadata.doc_name)

        ocr_time = time.perf_counter() - start_time
        workflow_job.processing_times["ocr"] = ocr_time
        
        return workflow_job, ocr_time
