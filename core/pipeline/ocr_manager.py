# PerfectOCR/core/coordinators/ocr_manager.py
import os
import numpy as np
import logging
import time
import json
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
from core.workers.ocr.paddle_wrapper import PaddleOCRWrapper
import cv2

logger = logging.getLogger(__name__)

class OCREngineManager:
    def __init__(self, config: Dict, stage_config: Dict, project_root: str):
        self.project_root = project_root
        self.padd_ocr = config
        paddle_specific_config = config.get('paddle', {})
        if paddle_specific_config:
            self.paddle = PaddleOCRWrapper(paddle_specific_config, self.project_root)
            
        else:
            self.paddle = None
            raise ValueError("PaddleOCR engine not enabled or configured for recognition.")

        self.num_workers = 1

    def _save_complete_ocr_results(self, output_data: Dict[str, Any], image_name: str) -> None:
        """
        Guarda los resultados completos del OCR en formato JSON si está habilitado.
        Incluye todos los metadatos y resultados estructurados.
        """
        if not self.manager_config.get('output_flag', {}).get('ocr_raw', False):
            return
            
        output_folder = self.manager_config.get('output_folder')
        if not output_folder:
            logger.warning("No se puede guardar resultados OCR porque 'output_folder' no está definido.")
            return
            
        try:
            os.makedirs(output_folder, exist_ok=True)
            base_name = image_name
            json_filename = f"{base_name}_ocr_results.json"
            json_path = os.path.join(output_folder, json_filename)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"OCREngineManager: Resultados OCR guardados en {json_path}")
        except Exception as e:
            logger.error(f"OCREngineManager: Error guardando resultados OCR: {e}")

    def _run_ocr_on_polygons(
        self, 
        preprocess_dict: Dict[str, Any]
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        
        start_time = time.perf_counter()
        
        if not self.paddle or self.paddle.engine is None:
            logger.error("PaddleOCR recognition engine not available.")
            return {"error": "PaddleOCR engine not available"}, 0.0

        # Extraer polígonos del diccionario
        polygons = preprocess_dict.get("polygons", {})
        if not polygons:
            logger.warning("No polygons were provided to OCREngineCoordinator.")
            return {"polygon_results": {}, "full_text": ""}, 0.0

        # Extraer solo las imágenes procesadas con sus IDs
        valid_polygons = []
        polygon_ids = []
        images_batch = []
        
        for polygon_id, polygon_data in polygons.items():
            processed_img = polygon_data.get("processed_img")
            if processed_img is not None and isinstance(processed_img, np.ndarray):
                valid_polygons.append(polygon_data)
                polygon_ids.append(polygon_id)
                # Convertir a BGR si es necesario
                if len(processed_img.shape) == 2:  # Grayscale
                    image_3d = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
                else:
                    image_3d = processed_img
                images_batch.append(image_3d)

        if not images_batch:
            logger.warning("No se encontraron polígonos con imágenes válidas para procesar.")
            return {"polygon_results": {}, "full_text": ""}, 0.0

        # Procesar lote con PaddleOCR
        batch_results = self.paddle.recognize_text_from_batch(images_batch)
        
        # Mapear resultados a IDs
        polygon_results = {}
        processed_count = 0
        for i, result in enumerate(batch_results):
            poly_id = polygon_ids[i]
            if result:
                polygon_results[poly_id] = result
                processed_count += 1
                # Actualizar el diccionario original: añadir texto y limpiar imagen
                polygons[poly_id]["text"] = result.get("text", "")
                polygons[poly_id]["confidence"] = result.get("confidence", 0.0)
                # Limpiar la imagen ya procesada
                polygons[poly_id]["processed_img"] = None
            else:
                logger.debug(f"El polígono {poly_id} no arrojó resultado en el lote.")

        logger.info(f"Lote procesado. Se obtuvieron resultados para {processed_count}/{len(valid_polygons)} polígonos.")

        output_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "ocr_engine": "paddleocr_recognition_only",
                "polygons_processed": processed_count,
                "polygons_total": len(polygons)
            },
            "polygon_results": polygon_results,
        }

        image_name = preprocess_dict.get("metadata", {}).get("doc_name", "unknown")
        self._save_complete_ocr_results(output_data, image_name)

        total_time = time.perf_counter() - start_time
        logger.info(f"Tiempo total del flujo OCR por lotes: {total_time:.3f}s")
        
        return output_data, total_time