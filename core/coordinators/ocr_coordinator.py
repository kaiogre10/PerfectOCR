# PerfectOCR/core/coordinators/ocr_coordinator.py
import os
import numpy as np
import logging
import time
from collections import defaultdict
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from core.workflow.ocr.paddle_wrapper import PaddleOCRWrapper
from core.workspace.utils.output_handlers import OutputHandler
import cv2

logger = logging.getLogger(__name__)

class OCREngineCoordinator:
    def __init__(self, config: Dict, project_root: str, output_flags: Dict[str, bool], workflow_config: Optional[Dict[str, Any]] = None):
        self.ocr_config_from_workflow = config
        self.project_root = project_root
        self.output_flags = output_flags
        self.workflow_config = workflow_config or {}
        self.json_output_handler = OutputHandler(config={"enabled_outputs": self.output_flags})
        
        # --- Inicialización del motor de RECONOCIMIENTO ---
        paddle_specific_config = self.ocr_config_from_workflow.get('paddleocr')
        if paddle_specific_config:
            self.paddle = PaddleOCRWrapper(paddle_specific_config, self.project_root)
        else:
            self.paddle = None
            raise ValueError("PaddleOCR engine not enabled or configured for recognition.")

        # Configuración conservadora para PaddleOCR (sensible a paralelización)
        self.num_workers = 1  # Procesamiento secuencial para estabilidad
        logger.info(f"OCREngineCoordinator using sequential processing for PaddleOCR stability.")

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
        
        # Guardar resultado detallado para cada polígono
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
                
                # Si el resultado no es None, mostrar más detalles
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

    def run_ocr_on_polygons(
        self, 
        polygons: List[Dict[str, Any]],
        image_file_name: str
    ) -> Tuple[Dict[str, Any], float]:
        
        start_time = time.perf_counter()
        
        if not self.paddle or self.paddle.engine is None:
            logger.error("PaddleOCR recognition engine not available.")
            return {"error": "PaddleOCR engine not available"}, 0.0

        if not polygons:
            logger.warning("No polygons were provided to OCREngineCoordinator.")
            return {"polygon_results": {}, "full_text": ""}, 0.0

        total_polygons = len(polygons)
        logger.info(f"=== INICIANDO OCR POR LOTES (BATCH) ===")
        logger.info(f"Agrupando {total_polygons} polígonos para procesamiento en lote...")

        # --- Preparación del Lote ---
        # 1. Filtrar polígonos que tienen una imagen válida.
        # 2. Mantener el ID para re-asociar los resultados después.
        valid_polygons = [p for p in polygons if p.get("processed_img") is not None and isinstance(p.get("processed_img"), np.ndarray)]
        if not valid_polygons:
            logger.warning("No se encontraron polígonos con imágenes válidas para procesar.")
            return {"polygon_results": {}, "full_text": ""}, 0.0
            
        images_batch = [p["processed_img"] for p in valid_polygons]
        polygon_ids = [p["polygon_id"] for p in valid_polygons]
        
        logger.info(f"Enviando lote de {len(images_batch)} imágenes a PaddleOCR...")

        images_batch_3d = []
        for image_2d in images_batch:
            image_3d = cv2.cvtColor(image_2d, cv2.COLOR_GRAY2BGR) 
            images_batch_3d.append(image_3d)

        # Y luego llamamos al OCR con el lote formateado correctamente
        batch_results = self.paddle.recognize_text_from_batch(images_batch_3d)
        self._save_paddle_raw_output(batch_results, polygon_ids, image_file_name)
        
        polygon_results = {}
        processed_count = 0
        for i, result in enumerate(batch_results):
            poly_id = polygon_ids[i]
            if result:
                polygon_results[poly_id] = result
                processed_count += 1
            else:
                logger.debug(f"El polígono {poly_id} no arrojó resultado en el lote.")

        logger.info(f"Lote procesado. Se obtuvieron resultados para {processed_count}/{len(valid_polygons)} polígonos.")

        # Consolidar el texto completo en el orden correcto si es necesario
        full_text = self._consolidate_full_text(polygons, polygon_results)

        # Empaquetar el resultado final
        output_data = {
            "metadata": {
                "image_file_name": image_file_name,
                "timestamp": datetime.now().isoformat(),
                "ocr_engine": "paddleocr_recognition_only",
                "polygons_processed": processed_count,
                "polygons_total": total_polygons
            },
            "polygon_results": polygon_results,
            "full_text": full_text
        }

        total_time = time.perf_counter() - start_time
        logger.info(f"Tiempo total del flujo OCR por lotes: {total_time:.3f}s")
        
        return output_data, total_time

    def _consolidate_full_text(self, original_polygons: List[Dict], ocr_results: Dict) -> str:
        """
        Reconstruye el texto completo del documento agrupando por línea y ordenando por la coordenada X.
        """
        lines = defaultdict(list)
        
        # Agrupar polígonos por line_id
        for poly in original_polygons:
            poly_id = poly.get("polygon_id")
            line_id = poly.get("line_id")
            result = ocr_results.get(poly_id)
            
            if result and result.get("text") and line_id is not None:
                try:
                    x_coordinate = poly['geometry']['bounding_box'][0]
                    lines[line_id].append((x_coordinate, result["text"]))
                except (KeyError, IndexError):
                    # Si no hay info geométrica, se añade al final sin orden específico
                    lines[line_id].append((float('inf'), result["text"]))

        # Ordenar cada línea por la coordenada X y luego unir el texto
        reconstructed_lines = []
        for line_id in sorted(lines.keys()):
            sorted_fragments = sorted(lines[line_id], key=lambda item: item[0])
            line_text = " ".join([text for x, text in sorted_fragments if text.strip()])
            reconstructed_lines.append(line_text)
        
        return "\n".join(reconstructed_lines)

    def validate_ocr_results(self, ocr_results: Optional[dict]) -> bool:
        """Valida si el resultado del OCR es aceptable."""
        if not isinstance(ocr_results, dict): 
            return False
        
        # El criterio de validación es simplemente si se reconoció texto en al menos un polígono.
        polygon_results = ocr_results.get("polygon_results", {})
        if not polygon_results:
            return False

        return any(res.get("text") for res in polygon_results.values())
