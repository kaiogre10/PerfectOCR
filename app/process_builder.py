# PerfectOCR/app/process_builder.py
import os
import logging
import cv2
import time
from typing import Dict, Optional, Any, List
from core.pipeline.input_manager import InputManager
from core.pipeline.preprocessing_manager import PreprocessingManager
from core.pipeline.ocr_manager import OCREngineManager

logger = logging.getLogger(__name__)

class ProcessingBuilder:
    """
    Director de Operaciones: Recibe a sus Jefes de Área ya entrenados y
    coordina el procesamiento técnico de una sola imagen.
    """
    def __init__(self, input_manager, preprocessing_manager, ocr_manager):
        self.input_manager = input_manager
        self.preprocessing_manager = preprocessing_manager
        self.ocr_manager = ocr_manager
        
    def _process_single_image(self, image_name: str) -> Optional[Dict[str, Any]]:
        """
        Procesa una sola imagen. El código interno de este método
        NO CAMBIA, ya que sigue usando self.polygon_manager, etc.
        """
        workflow_start = time.perf_counter()
        processing_times_summary: Dict[str, float] = {}

        try:
            # FASE 1: Cargar imagen y obtener polígonos
            phase1_start = time.perf_counter()
                
            extracted_polygons, time_poly = self.input_manager._generate_polygons()
            
            polygons_to_bin = self.input_manager._get_polygons_to_binarize()
            
            phase1_time = time.perf_counter() - phase1_start
            processing_times_summary["1_polygons"] = round(time_poly, 4)
            logger.info(f"Polígonos generados en: {phase1_time:.3f}s")
            
            if extracted_polygons is None:
                logger.critical("No se pudieron generar polígonos válidos.")
                return None
            
            # FASE 2: Preprocesamiento
            phase2_start = time.perf_counter()
            preprocess_dict, total_duration = self.preprocessing_manager._apply_preprocessing_pipelines(
                extracted_polygons, polygons_to_bin
            )
            phase2_time = time.perf_counter() - phase2_start
            processing_times_summary["2_preprocesamiento"] = round(total_duration, 4)
            logger.info(f"Preprocesamiento: {phase2_time:.3f}s")

            # Validar que el preprocesamiento fue exitoso
            if preprocess_dict is None:
                logger.critical("El preprocesamiento falló.")
                return None

            # FASE 3: OCR
            phase3_start = time.perf_counter()
            ocr_raw, time_ocr = self.ocr_manager._run_ocr_on_polygons(preprocess_dict)
            phase3_time = time.perf_counter() - phase3_start
            processing_times_summary["3_ocr"] = round(time_ocr, 4)
            logger.info(f"OCR: {time_ocr:.3f}s")
            			
            # RESULTADO FINAL
            total_workflow_time = time.perf_counter() - workflow_start
            processing_times_summary["total_workflow"] = round(total_workflow_time, 4)
            
            result = {
                "image_name,": image_name,
                "status": "success",
                "processing_times": processing_times_summary,
                "ocr_results": ocr_raw,
                "total_time": total_workflow_time
            }
            
            logger.info(f"Procesamiento completado: {total_workflow_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error procesando {image_name}: {e}")
            return {
                "image_name": image_name,
                "status": "error",
                "error": str(e),
                "processing_times": processing_times_summary
            }
