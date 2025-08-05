# PerfectOCR/app/process_builder.py
import os
import logging
import cv2
import time
from typing import Dict, Optional, Any, List
from core.pipeline.input_stager import InputManager
from core.pipeline.preprocessing_stager import PreprocessingManager
from core.pipeline.ocr_stager import OCREngineManager
from core.domain.workflow_job import WorkflowJob, ProcessingStage, ProcessingStatus

logger = logging.getLogger(__name__)

class ProcessingBuilder:
    """
    Director de Operaciones: Recibe a sus Jefes de Área ya entrenados y
    coordina el procesamiento técnico de una sola imagen.
    """
    def __init__(self, input_manager: InputManager, preprocessing_manager: PreprocessingManager, ocr_manager: OCREngineManager):
        self.input_manager = input_manager
        self.preprocessing_manager = preprocessing_manager
        self.ocr_manager = ocr_manager
        
    def _process_single_image(self, image_name: str) -> Optional[WorkflowJob]:
        """
        Procesa una sola imagen. El ProcessingBuilder SOLO COORDINA, no convierte.
        """
        logger.info(f"[ProcessingBuilder] Iniciando procesamiento de imagen: {image_name}")
        workflow_start = time.perf_counter()

        try:
            # FASE 1: Cargar imagen y obtener polígonos (InputManager devuelve WorkflowJob)
            workflow_job, time_poly = self.input_manager._generate_polygons()
            
            if workflow_job is None:
                logger.error("[ProcessingBuilder] No se pudo generar WorkflowJob desde InputManager")
                error_job = WorkflowJob(job_id=f"error_{image_name}_{int(time.time())}", status=ProcessingStatus.FAILED)
                error_job.add_error("Fallo en la fase de generación de polígonos.")
                return error_job

            # FASE 2: Preprocesamiento (PreprocessingManager modifica WorkflowJob)
            polygons_to_bin = self.input_manager._get_polygons_to_binarize()
            workflow_job, prep_time = self.preprocessing_manager._apply_preprocessing_pipelines(workflow_job, polygons_to_bin)

            # FASE 3: OCR (OCREngineManager modifica WorkflowJob)
            if workflow_job:
                workflow_job, ocr_time = self.ocr_manager._run_ocr_on_polygons(workflow_job)
            
            # RESULTADO FINAL
            if workflow_job:
                total_workflow_time = time.perf_counter() - workflow_start
                workflow_job.processing_times["total_workflow"] = round(total_workflow_time, 4)
                workflow_job.status = ProcessingStatus.COMPLETED
                logger.info(f"[ProcessingBuilder] Procesamiento completado en {total_workflow_time:.3f}s")
            
            return workflow_job
            
        except Exception as e:
            logger.error(f"[ProcessingBuilder] Error fatal procesando {image_name}: {e}", exc_info=True)
            error_job = WorkflowJob(job_id=f"error_{image_name}_{int(time.time())}", status=ProcessingStatus.FAILED)
            error_job.add_error(f"Error fatal en ProcessingBuilder: {e}")
            return error_job
