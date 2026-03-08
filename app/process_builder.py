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
        
    def process_single_image(self) -> Optional[str]:
        """
        Procesa una sola imagen usando el método execute() uniforme de cada stager.
        """
        try:
            # 1. El main_builder ya mutó el context compartido con la info de la nueva imagen
            
            # 2. Formatear/Reiniciar el DataFormatter para la nueva tarea
            if hasattr(self.manager, 'reset'):
                self.manager.reset()
            
            # 3. Flujo normal (los stagers ya tienen la referencia al shared_context)
            # Solo se ejecutan los stagers que fueron configurados y devueltos por la factory
            
            # Input siempre es obligatorio
            current_result, _ = self.input_stager.execute(self.manager)
            if not current_result: 
                return None
            
            if self.preprocessing_stager:
                current_result, _ = self.preprocessing_stager.execute(current_result)
                if not current_result: 
                    return None
            
            if self.ocr_stager:
                current_result, _ = self.ocr_stager.execute(current_result)
                if not current_result: 
                    return None
            
            if self.vectorization_stager:
                current_result, _ = self.vectorization_stager.execute(current_result)
                if not current_result: 
                    return None
            
            # 4. Dependiendo de si se corrió vectorización u ocr, el output final se procesa aquí
            if hasattr(self.manager, 'format'):
                db_path = self.manager.format(current_result)
            else:
                # Si format no está implementado aún o devuelve None, omitimos
                db_path = None
            
            return db_path
            
        except Exception as e:
            logger.error(f"Error procesando imagen: {e}", exc_info=True)
            return None