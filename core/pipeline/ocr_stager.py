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
        
    def run_ocr_on_polygons(self, manager: DataFormatter) -> Tuple[Optional[DataFormatter], float]:
        start_time = time.perf_counter()
        
        # Obtener datos (igual que preprocessing)
        polygons = manager.get_polygons_with_cropped_img()
        
        # DEBUG: Ver qué contiene polygons
        logger.info(f"[OCREngineManager] Polígonos obtenidos: {len(polygons)}")
        for poly_id, poly_data in list(polygons.items())[:3]:  # Solo los primeros 3
            cropped_img = poly_data.get("cropped_img")
            logger.info(f"[OCREngineManager] {poly_id}: cropped_img type={type(cropped_img)}, shape={getattr(cropped_img, 'shape', 'N/A')}")
        
        # Preparar batch (igual que preprocessing pero optimizado para OCR)
        image_list = []
        polygon_ids = []
        
        for poly_id, poly_data in polygons.items():
            cropped_img = poly_data.get("cropped_img")
            if cropped_img is not None:
                # Convertir a np.ndarray si es necesario
                if isinstance(cropped_img, list):
                    cropped_img = np.array(cropped_img)
                
                # Validar que la imagen sea procesable
                if hasattr(cropped_img, 'shape') and len(cropped_img.shape) >= 2:
                    if min(cropped_img.shape[:2]) > 0:  # Dimensiones válidas
                        # ✅ CONVERTIR A 3 CANALES para PaddleOCR
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
        batch_results = self.paddle.recognize_text_from_batch(image_list)
        
        # Actualizar resultados (igual que preprocessing)
        processed_count = 0
        if batch_results:
            for idx, res in enumerate(batch_results):
                if idx < len(polygon_ids):
                    poly_id = polygon_ids[idx]
                    if res and manager.workflow_dict and poly_id in manager.workflow_dict["polygons"]:
                        manager.workflow_dict["polygons"][poly_id]["ocr_text"] = res.get("text", "")
                        manager.workflow_dict["polygons"][poly_id]["ocr_confidence"] = res.get("confidence")
                        processed_count += 1
        
        logger.info(f"[OCREngineManager] Batch OCR completado. {processed_count}/{len(image_list)} polígonos procesados.")
        
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