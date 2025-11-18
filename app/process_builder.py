# PerfectOCR/app/process_builder.py
import time
import logging
from typing import Optional
from core.pipeline.image_preparation_stager import ImagePreparationStager
from core.pipeline.preprocessing_stager import PreprocessingStager
from core.pipeline.ocr_stager import OCRStager
from core.pipeline.vectorization_stager import VectorizationStager
from core.domain.data_formatter import DataFormatter

logger = logging.getLogger(__name__)

class ProcessingBuilder:
    """Director de Operaciones: Recibe a sus Jefes de Área ya entrenados ycoordina el procesamiento técnico de una sola imagen."""    
    def __init__(self, input_stager: ImagePreparationStager, preprocessing_stager: Optional[PreprocessingStager], ocr_stager: Optional[OCRStager], vectorization_stager: Optional[VectorizationStager] ,manager: DataFormatter):
        self.manager = manager
        self.input_stager = input_stager
        self.preprocessing_stager = preprocessing_stager
        self.ocr_stager = ocr_stager
        self.vectorization_stager = vectorization_stager
        
    def process_single_image(self) -> Optional[DataFormatter]:
        """
        Procesa una sola imagen usando el método execute() uniforme de cada stager.
        """
        try:
            workflow_start = time.perf_counter()
            manager = DataFormatter()
            
            # FASE 1: Preparación de imagen (usa execute() del AbstractStager)
            manager, time_poly = self.input_stager.execute(manager)
            if manager is None:
                logger.error("Fallo en fase de preparación")
                return None
            logger.debug(f"Fase de preparación completada en: {time_poly:.6f}s")

            # FASE 2: Preprocesamiento (usa execute() del AbstractStager)
            if self.preprocessing_stager:
                manager, elapsed = self.preprocessing_stager.execute(manager)
                if manager is None:
                    logger.error("Fallo en preprocesamiento")
                    return None
                logger.debug(f"Fase de preprocesamiento completada en: {elapsed:.6f}s")

            # FASE 3: OCR (usa execute() del AbstractStager)
            if self.ocr_stager:
                manager, ocr_time = self.ocr_stager.execute(manager)
                if manager is None:
                    logger.error("Fallo en OCR")
                    return None
                logger.debug(f"OCR completado en: {ocr_time:.6f}s")
                    
            # FASE 4: Vectorización (usa execute() del AbstractStager)
            if self.vectorization_stager:
                manager, vect_time = self.vectorization_stager.execute(manager)
                if manager is None:
                    logger.error("Fallo en vectorización")
                    return None
                logger.debug(f"Vectorización completada en: {vect_time:.6f}s")
            
            total_workflow_time = time.perf_counter() - workflow_start
            logger.debug(f"Procesamiento completado en {total_workflow_time:.6f}s")

            return manager
            
        except Exception as e:
            logger.error(f"Error fatal procesando la imagen: {e}", exc_info=True)
            return None