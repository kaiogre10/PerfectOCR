# PerfectOCR/core/coordinators/ocr_manager.py
import os
import numpy as np
import logging
import time
from collections import defaultdict
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from core.workers.ocr.paddle_wrapper import PaddleOCRWrapper
import cv2

logger = logging.getLogger(__name__)

class OCREngineManager:
    def __init__(self, config: Dict, project_root: str, output_flags: Dict[str, bool], workflow_config: Optional[Dict[str, Any]] = None):
        self.ocr_config_from_workflow = config
        self.project_root = project_root
        self.output_flags = output_flags
        self.workflow_config = workflow_config or {}
        paddle_specific_config = self.ocr_config_from_workflow.get('paddleocr')
        if paddle_specific_config:
            self.paddle = PaddleOCRWrapper(paddle_specific_config, self.project_root)
            
        else:
            self.paddle = None
            raise ValueError("PaddleOCR engine not enabled or configured for recognition.")

        self.num_workers = 1

    def _save_paddle_raw_output(self, batch_results: List, polygon_ids: List[str], image_file_name: str) -> None:
        """
        Guarda la salida raw completa de PaddleOCR para cada polígono.
        Incluye toda la información que devuelve PaddleOCR, no solo el texto final.
        """
        if not self.output_flags.get('ocr_debug', False):
            return
            
        output_folder = self.workflow_config.get('output_folder')
        if not output_folder:
            logger.error("No se puede guardar debug de OCR porque 'output_folder' no está definido.")
            return
            
        base_name = os.path.splitext(image_file_name)[0]
        debug_folder = os.path.join(output_folder, f"{base_name}_paddle_debug")
        os.makedirs(debug_folder, exist_ok=True)
        

        batch_info = {
            "timestamp": datetime.now().isoformat(),
            "total_polygons": len(polygon_ids),
            "batch_results_count": len(batch_results),
            "image_file_name": image_file_name
        }
        
        batch_info_path = os.path.join(debug_folder, "batch_info.txt")
        with open(batch_info_path, 'w', encoding='utf-8') as f:
            f.write("=== INFORMACIÓN DEL BATCH ===\n")
            for key, value in batch_info.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
        
        for i, (polygon_id, result) in enumerate(zip(polygon_ids, batch_results)):
            debug_filename = f"polygon_{polygon_id}_paddle_raw.txt"
            debug_path = os.path.join(debug_folder, debug_filename)
            
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write(f"=== POLÍGONO {polygon_id} ===\n")
                f.write(f"Índice en batch: {i}\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Resultado raw de PaddleOCR:\n")
                f.write(f"{result}\n")
                f.write(f"Tipo de resultado: {type(result)}\n")
                f.write(f"Longitud del resultado: {len(str(result))}\n")
                f.write("\n")
                
                if result is not None:
                    f.write("=== ANÁLISIS DETALLADO ===\n")
                    f.write(f"¿Es lista?: {isinstance(result, list)}\n")
                    f.write(f"¿Es diccionario?: {isinstance(result, dict)}\n")
                    if isinstance(result, list):
                        f.write(f"Longitud de la lista: {len(result)}\n")
                        for j, item in enumerate(result):
                            f.write(f"  Item {j}: {item} (tipo: {type(item)})\n")
                    elif isinstance(result, dict):
                        for key, value in result.items():
                            f.write(f"  {key}: {value} (tipo: {type(value)})\n")
                else:
                    f.write("RESULTADO: None (PaddleOCR no devolvió nada)\n")
        
        logger.info(f"Debug de PaddleOCR guardado en: {debug_folder}")

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

        total_time = time.perf_counter() - start_time
        logger.info(f"Tiempo total del flujo OCR por lotes: {total_time:.3f}s")
        
        return output_data, total_time