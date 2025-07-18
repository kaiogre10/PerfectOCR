# PerfectOCR/core/coordinators/ocr_coordinator.py
import os
import numpy as np
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from core.workflow.ocr.paddle_wrapper import PaddleOCRWrapper
from core.workspace.utils.output_handlers import OutputHandler

logger = logging.getLogger(__name__)

class OCREngineCoordinator:
    def __init__(self, config: Dict, project_root: str, output_flags: Dict[str, bool], workflow_config: Dict[str, Any] = None):
        self.ocr_config_from_workflow = config
        self.project_root = project_root
        self.output_flags = output_flags
        self.workflow_config = workflow_config or {}
        self.json_output_handler = OutputHandler(config={"enabled_outputs": self.output_flags})
        
        # Calcular workers óptimos dinámicamente para PaddleOCR
        cpu_count = os.cpu_count() or 4
        self.num_workers = min(6, cpu_count)

        # Inicialización directa de PaddleOCR
        paddle_specific_config = self.ocr_config_from_workflow.get('paddleocr')
        if paddle_specific_config:
            singleton_start = time.perf_counter()
            logger.info("Inicializando PaddleOCR...")
            self.paddle = PaddleOCRWrapper(paddle_specific_config, self.project_root)
            singleton_time = time.perf_counter() - singleton_start
            logger.info(f"PaddleOCR inicializado: {singleton_time:.3f}s")
        else:
            self.paddle = None
        
        if not self.paddle:
            raise ValueError("PaddleOCR engine not enabled or configured")

    def _run_paddle_task(self, img, fname=None):
        if self.paddle:
            return self.paddle.extract_detailed_line_data(img, fname)
        else:
            return {
                "ocr_engine": "paddleocr",
                "processing_time_seconds": 0.0,
                "recognized_text": {"lines": [], "words": []},
                "error": "PaddleOCR not enabled"
            }

    def run_ocr_parallel(self, preprocessed_images: Dict[str, np.ndarray], 
                        image_file_name: Optional[str] = None, folder_origin: Optional[str] = "unknown_folder", 
                        image_pil_mode: Optional[str] = "unknown_mode") -> Tuple[Dict[str, Any], float]:
        
        start_time = time.perf_counter()
        
        # DEPURACIÓN CORREGIDA: Verificar qué imágenes llegan
        logger.debug(f"OCR recibió imágenes para: {list(preprocessed_images.keys())}")
        for engine, img in preprocessed_images.items():
            if img is not None and hasattr(img, 'shape'):
                logger.debug(f"{engine}: imagen {img.shape} dtype={img.dtype}")
            elif img is not None:
                logger.warning(f"{engine}: objeto recibido es {type(img)} (esperaba numpy array)")
            else:
                logger.warning(f"{engine}: imagen es None!")
        
        folder_origin = self.ocr_config_from_workflow.get('default_folder_origin', "unknown_folder")
        image_pil_mode = self.ocr_config_from_workflow.get('default_image_pil_mode', "unknown_mode")
        
        paddle_image = preprocessed_images.get('paddleocr') if self.paddle else None
                    
        if self.paddle and paddle_image is None:
            logger.error("PaddleOCR habilitado pero imagen faltante") 
            return {"error": "Missing PaddleOCR preprocessed image"}, time.perf_counter() - start_time

        elif paddle_image is not None:
            page_dims = {"width": paddle_image.shape[1], "height": paddle_image.shape[0]}
        else:
            page_dims = {"width": 0, "height": 0}

        # Ejecutar solo PaddleOCR
        execution_start = time.perf_counter()
        padd_result_payload = None
        
        if self.paddle:
            padd_result_payload = self._run_paddle_task(paddle_image, image_file_name)

        execution_time = time.perf_counter() - execution_start

        # Consolidar resultados
        output_data = self._consolidate_ocr_results(
            padd_result_payload, 
            image_file_name, folder_origin, image_pil_mode, page_dims
        )

        total_time = time.perf_counter() - start_time
        
        return output_data, total_time

    def _consolidate_ocr_results(self, padd_result_payload: Optional[Dict],
                               image_file_name: str, folder_origin: str, image_pil_mode: str, 
                               page_dims: Dict) -> Dict[str, Any]:
        
        output_data = {
            "metadata": {
                "image": image_file_name,
                "folder_origin": folder_origin,
                "image_pil_mode": image_pil_mode,
                "timestamp": datetime.now().isoformat(),
                "page_dimensions": page_dims,
                "enabled_engines": {
                    "paddleocr": self.paddle is not None
                },
                "processing_time_seconds": {
                    "paddleocr": padd_result_payload.get("processing_time_seconds", 0.0) if padd_result_payload else 0.0
                },
                "overall_confidence": {}
            },
            "ocr_raw_results": {},
            "visual_output": {"paddleocr_text": ""}
        }
        
        # PaddleOCR
        if padd_result_payload and "error" not in padd_result_payload:
            output_data["ocr_raw_results"]["paddleocr"] = {
                "lines": padd_result_payload.get("recognized_text", {}).get("lines", []),
                "full_text": padd_result_payload.get("recognized_text", {}).get("full_text", ""),
                "words": padd_result_payload.get("recognized_text", {}).get("words", [])
            }
            output_data["metadata"]["overall_confidence"]["paddleocr_lines_avg"] = padd_result_payload.get("overall_confidence_avg_lines")
            output_data["visual_output"]["paddleocr_text"] = output_data["ocr_raw_results"]["paddleocr"]["full_text"]
        elif padd_result_payload and "error" in padd_result_payload:
            output_data["ocr_raw_results"]["paddleocr"] = {"error": padd_result_payload.get("error")}
        
        # Guardar JSON
        ocr_json_path = None
        if self.output_flags.get('ocr_raw', False) and image_file_name:
            output_dir = self.workflow_config.get('output_folder', os.path.join(self.project_root, "output"))
            base_name = os.path.splitext(os.path.basename(image_file_name))[0]
            ocr_json_path = self.json_output_handler.save(
                data=output_data,
                output_dir=output_dir,
                file_name_with_extension=f"{base_name}_ocr_raw_results.json"
            )
            
        output_data["ocr_raw_json_path"] = ocr_json_path
        return output_data

    def validate_ocr_results(self, ocr_results: Optional[dict], filename: str) -> bool:
        """Valida resultados OCR"""
        if not isinstance(ocr_results, dict): 
            return False
        
        ocr_raw_results = ocr_results.get("ocr_raw_results", {})
        enabled_engines = ocr_results.get("metadata", {}).get("enabled_engines", {})
        
        has_valid_text = False
        
        if enabled_engines.get("paddleocr", False):
            paddle_data = ocr_raw_results.get("paddleocr", {})
            if "error" not in paddle_data and len(paddle_data.get("lines", [])) > 0:
                has_valid_text = True
        
        return has_valid_text