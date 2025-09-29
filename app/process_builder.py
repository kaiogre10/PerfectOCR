# PerfectOCR/app/process_builder.py
import time
import logging
from typing import Optional, Dict
from core.pipeline.image_preparation_stager import ImagePreparationStager
from core.pipeline.preprocessing_stager import PreprocessingStager
from core.pipeline.ocr_stager import OCRStager
from core.pipeline.vectorization_stager import VectorizationStager
from core.domain.data_formatter import DataFormatter

logger = logging.getLogger(__name__)

class ProcessingBuilder:
    """Director de Operaciones: Recibe a sus Jefes de Área ya entrenados ycoordina el procesamiento técnico de una sola imagen."""
    
    def __init__(self, config: Dict[str, str] , input_stager: ImagePreparationStager, preprocessing_stager: PreprocessingStager, ocr_stager: OCRStager, vectorization_stager: VectorizationStager ,manager: DataFormatter):
        self.config = config
        self.manager = manager
        self.input_stager = input_stager
        self.preprocessing_stager = preprocessing_stager
        self.ocr_stager = ocr_stager
        self.vectorization_stager = vectorization_stager
        
    def process_single_image(self) -> Optional[str]:
        """
        Procesa una sola imagen. El ProcessingBuilder SOLO COORDINA, no convierte.
        """
        try:
            # FASE 1: Cargar imagen, obtener polígonos y liberar full_img
            workflow_start = time.perf_counter()
            manager = DataFormatter()
            manager, time_poly = self.input_stager.generate_polygons(manager)
            
            if manager is None:
                logger.error("[ProcessingBuilder] No se pudo procesar la Fase de preparación.")
                return None
            
            logger.info(f"Fase de preparación completada en: {time_poly:.6f}s")

            # Fase 2: Corregir y preparar la imagen para el OCR
            manager, elapsed = self.preprocessing_stager.apply_preprocessing_pipelines(manager)
            
            logger.info(f"Fase de preprocesamiento completada en: {elapsed:.6f}s")

            # # FASE 3: OCR (modifica el manager y libera los recortes)
            if manager:
                manager, ocr_time = self.ocr_stager.run_ocr_on_polygons(manager)
                if manager is None:
                    logger.error("[ProcessingBuilder] No se pudo generar WorkflowDict desde OCR")
                else:
                    logger.info(f"OCR time: {ocr_time:.6f}s")
                    
            # Fase 4: Vectorización y Tokenización
            if manager:
                manager, vect_time = self.vectorization_stager.vectorize_results(manager)
                if manager is None:
                    logger.error("[ProcessingBuilder] No se pudieron generar vectores para el WorkflowDict")
                else:
                    vect_total = vect_time  # vect_time ya es la duración
                    logger.info(f"Vectorización time {vect_total:.6f}s")
            
            total_workflow_time = time.perf_counter() - workflow_start
            logger.info(f"[ProcessingBuilder] Procesamiento completado en {total_workflow_time:.6f}s")

            # data_base_path: str = self.config.get("db_path", "")
            # db_path: str = manager.to_db_payload(data_base_path)

            return manager
            
        except Exception as e:
            logger.error(f"[ProcessingBuilder] Error fatal procesando la imagen: {e}", exc_info=True)
        return None