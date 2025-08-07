# PerfectOCR/app/process_builder.py
import logging
import time
from typing import Dict, Optional, Any
from core.pipeline.input_stager import InputStager
from core.pipeline.preprocessing_stager import PreprocessingStager
from core.pipeline.ocr_stager import OCRStager
from core.domain.workflow_job import WorkflowJob, ProcessingStatus

logger = logging.getLogger(__name__)

class ProcessingBuilder:
    """
    Director de Operaciones: Recibe a sus Jefes de Área ya entrenados y
    coordina el procesamiento técnico de una sola imagen.
    """
    def __init__(self, input_stager: InputStager, preprocessing_stager: PreprocessingStager, ocr_stager: OCRStager):
        self.input_stager = input_stager
        self.preprocessing_stager = preprocessing_stager
        self.ocr_stager = ocr_stager
        
    def process_single_image(self, image_data: Dict[str, Any]) -> Optional[WorkflowJob]:
        """
        Procesa una sola imagen. El ProcessingBuilder SOLO COORDINA, no convierte.
        """
        image_name = image_data.get('name')
        logger.info(f"[ProcessingBuilder] Iniciando procesamiento de imagen: {image_name}")
        
        try:
            # FASE 1: Cargar imagen y obtener polígonos (InputManager devuelve WorkflowJob)
            workflow_start = time.perf_counter()
            workflow_job, time_poly = self.input_stager.generate_polygons()
            
            if workflow_job is None:
                logger.error("[ProcessingBuilder] No se pudo generar WorkflowJob desde InputStager")
                error_job = WorkflowJob(job_id=f"error_{image_name}_{int(time.time())}", status=ProcessingStatus.FAILED)
                error_job.add_error("Fallo en la fase de generación de polígonos.")
                return error_job

            logger.info(f"Poligonal time: {time_poly:4f}")

            # FASE 2: Preprocesamiento (PreprocessingManager modifica WorkflowJob)

            workflow_job, prep_time = self.preprocessing_stager.apply_preprocessing_pipelines(workflow_job)
            
            if workflow_job is None:
                logger.error("[ProcessingBuilder] No se pudo generar WorkflowJob desde Preprocesing")
                error_job = WorkflowJob(job_id=f"error_{image_name}_{int(time.time())}", status=ProcessingStatus.FAILED)
                error_job.add_error("Fallo en la fase de preprocesamiento.")
                return error_job
            else:

                logger.info(f"Preprocessing time: {prep_time}")


            # FASE 3: OCR (OCREngineManager modifica WorkflowJob)
            ocr_initime = time.perf_counter()
            if workflow_job:
                workflow_job, ocr_time = self.ocr_stager.run_ocr_on_polygons(workflow_job)
                if workflow_job is None:
                    logger.error("[ProcessingBuilder] No se pudo generar WorkflowJob desde OCR")
                    error_job = WorkflowJob(job_id=f"error_{image_name}_{int(time.time())}", status=ProcessingStatus.FAILED)
                    error_job.add_error("Fallo en la fase de OCR.")
                    return error_job
                else:
                    ocr_total = ocr_initime - ocr_time 
                    logger.info(f"OCR time: {ocr_total}")
                    
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
