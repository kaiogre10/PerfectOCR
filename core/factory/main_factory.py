# core/workers/factory/main_factory.py
from typing import Dict, Any, Optional
from core.workers.image_preparation.image_preparation_factory import ImagePreparationFactory
from core.workers.preprocessing.preprocessing_factory import PreprocessingFactory
from core.workers.ocr.ocr_factory import OCRFactory
from core.workers.vectorial_transformation.vectorizing_factory import VectorizingFactory
import logging

logger = logging.getLogger(__name__)

class MainFactory:
    """Factory universal que coordina todas las factories de módulos."""
    def __init__(self, modules_config: Dict[str, Any], project_root: str):
        self.modules_config = modules_config
        self.project_root = project_root
        
        image_preparation_config = self.modules_config.get("image_preparation", {})
        preprocessing_config = self.modules_config.get('preprocessing', {})
        ocr_config = self.modules_config.get('ocr', {})
        vectorizing_config = self.modules_config.get('vectorization', {})

        # Registro de fábricas por nombre de módulo
        self.module_factories: Dict[str, Any] = {
            "image_preparation": ImagePreparationFactory(
                image_preparation_config,
                project_root
            ) if image_preparation_config else None,
            "preprocessing": PreprocessingFactory(
                preprocessing_config,
                project_root
            ) if preprocessing_config else None,
            "ocr": OCRFactory(
                ocr_config,
                project_root
            ) if ocr_config else None,
            "vectorization": VectorizingFactory(
                vectorizing_config,
                project_root
            ) if vectorizing_config else None,
        }

    def get_image_preparation_factory(self) -> Optional[ImagePreparationFactory]:
        return self.module_factories["image_preparation"]

    def get_preprocessing_factory(self) -> Optional[PreprocessingFactory]:
        return self.module_factories["preprocessing"]
        
    def get_ocr_factory(self) -> Optional[OCRFactory]:
        return self.module_factories["ocr"]
        
    def get_vectorizing_factory(self) -> Optional[VectorizingFactory]:
        return self.module_factories["vectorization"]
