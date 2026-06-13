# PerfectOCR/app/process_builder.py
import logging
from typing import Optional, Dict, Any
from core.pipeline.image_preparation_stager import ImagePreparationStager
from core.pipeline.preprocessing_stager import PreprocessingStager
from core.pipeline.ocr_stager import OCRStager
from core.pipeline.vectorization_stager import VectorizationStager
from core.domain.data_formatter import DataFormatter
import services.storage_service as storage_service

logger = logging.getLogger(__name__)

class ProcessingBuilder:
    """Director de Operaciones: Recibe a sus Jefes de Área ya entrenados y coordina el procesamiento técnico de una sola imagen."""    
    def __init__(self, input_stager: Optional[ImagePreparationStager], preprocessing_stager: Optional[PreprocessingStager], ocr_stager: Optional[OCRStager], vectorization_stager: Optional[VectorizationStager], logs_config: Dict[str, Any]):
        self.input_stager = input_stager
        self.preprocessing_stager = preprocessing_stager
        self.ocr_stager = ocr_stager
        self.vectorization_stager = vectorization_stager
        self.logs_config = logs_config
        self.time_stages_log = logs_config.get("time_stages_log")
        self.time_worker_log = logs_config.get("time_worker_log")
        
    def process_single_image(self, image_data: Dict[str, Any]) -> Optional[Any]:
        """
        Procesa una sola imagen usando el método execute() uniforme de cada stager.
        Recibe image_data para configurar el contexto de esta ejecución específica.
        Devuelve Direcciones en memoria de los datos generados
        """
        try:    
            if self.input_stager is None:
                logger.warning("No hay Modulo de carga de imagen, acabando")
                return None

            # Crear instancia fresca de DataFormatter para esta imagen
            manager = DataFormatter(self.logs_config)
            # Crear contexto para esta ejecución
            context: Dict[str, Any] = {
                "image_data": image_data,
                "time_worker_log": self.time_worker_log
            }
            
            # FASE 1: Preparación de imagen (usa execute() del AbstractStager)
            # Pasamos contexto que incluye image_data para que el ImageLoader sepa qué cargar
            manager, time_poly = self.input_stager.execute(manager, context)
            if manager is None:
                return None
            if self.time_stages_log:
                logger.info(f"Fase de preparación completada en: {time_poly:.6f}s")

            # FASE 2: Preprocesamiento (usa execute() del AbstractStager)
            if self.preprocessing_stager is not None:
                manager, elapsed = self.preprocessing_stager.execute(manager, context)
                if manager is None:
                    return None

                if self.time_stages_log:
                    logger.info(f"Fase de preprocesamiento completada en: {elapsed:.6f}s")

            # FASE 3: OCR (usa execute() del AbstractStager)
            if self.ocr_stager is not None:
                manager, ocr_time = self.ocr_stager.execute(manager, context)
                if manager is None:
                    return None
                if self.time_stages_log:
                    logger.info(f"OCR completado en: {ocr_time:.6f}s")

            # FASE 4: Vectorización (usa execute() del AbstractStager)
            if self.vectorization_stager is not None:
                manager, vect_time = self.vectorization_stager.execute(manager, context)
                if manager is None:
                    return None
                if self.time_stages_log:
                    logger.info(f"Vectorización completada en: {vect_time:.6f}s")

            img_results = manager.get_final_data()
            storage_service.storage_data(img_results)
            return img_results
            
        except Exception as e:
            logger.error(f"Error fatal procesando la imagen: '{e}'", exc_info=True)
        return None
