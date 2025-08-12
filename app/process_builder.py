# PerfectOCR/app/process_builder.py
import logging
import time
from typing import Optional
from core.pipeline.input_stager import InputStager
from core.pipeline.preprocessing_stager import PreprocessingStager
#from core.pipeline.ocr_stager import OCRStager
from core.domain.data_formatter import DataFormatter

logger = logging.getLogger(__name__)

class ProcessingBuilder:
    """
    Director de Operaciones: Recibe a sus Jefes de Área ya entrenados y
    coordina el procesamiento técnico de una sola imagen.
    """
    def __init__(self, input_stager: InputStager, preprocessing_stager: PreprocessingStager, manager: DataFormatter): #ocr_stager: OCRStager
        self.manager = manager
        self.input_stager = input_stager
        self.preprocessing_stager = preprocessing_stager
        #self.ocr_stager = ocr_stager
        
    def process_single_image(self) -> Optional[DataFormatter]:
        """
        Procesa una sola imagen. El ProcessingBuilder SOLO COORDINA, no convierte.
        """
        try:
            # FASE 1: Cargar imagen, obtener polígonos y liberar full_img
            workflow_start = time.perf_counter()
            manager = DataFormatter()
            
            manager, time_poly = self.input_stager.generate_polygons(manager)
            
            if manager is None:
                logger.error("[ProcessingBuilder] No se pudo procesar la fase de entrada.")
                return None
            
            logger.info(f"Fase de entrada completada en: {time_poly:.4f}s")

            # Fase 2: Corregir y preparar la imagen para el OCR
            manager, time_poly = self.preprocessing_stager.apply_preprocessing_pipelines(manager)
            
            logger.info(f"Fase de preprocesamiento completada en: {time_poly:.4f}s")

            # FASE 3: OCR (modifica el manager y libera los recortes)
            # ocr_initime = time.perf_counter()
            # if workflow_job:
            #     workflow_job, ocr_time = self.ocr_stager.run_ocr_on_polygons(workflow_job)
            #     if workflow_job is None:
            #         logger.error("[ProcessingBuilder] No se pudo generar WorkflowJob desde OCR")
            #         error_job.add_error("Fallo en la fase de OCR.")
            #         return error_job
            #     else:
            #         ocr_total = ocr_initime - ocr_time 
            #         logger.info(f"OCR time: {ocr_total}")
                    
            # RESULTADO FINAL
            total_workflow_time = time.perf_counter() - workflow_start
            # Aquí podrías añadir el tiempo total al dict del manager si quieres
            # processed_manager.set_value(...) 
            logger.info(f"[ProcessingBuilder] Procesamiento completado en {total_workflow_time:.3f}s")
            
            return manager
            
        except Exception as e:
            logger.error(f"[ProcessingBuilder] Error fatal procesando la imagen: {e}", exc_info=True)
            # Podrías crear un manager con el error si es necesario
        return None