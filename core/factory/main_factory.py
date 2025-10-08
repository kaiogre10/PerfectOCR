# core/workers/factory/main_factory.py
from typing import Dict, Any
from core.workers.image_preparation.image_preparation_factory import ImagePreparationFactory
from core.workers.preprocessing.preprocessing_factory import PreprocessingFactory
from core.workers.ocr.ocr_factory import OCRFactory
from core.workers.vectorial_transformation.vectorizing_factory import VectorizingFactory

class MainFactory:
    """Factory universal que coordina todas las factories de módulos."""

    def __init__(self, modules_config: Dict[str, Any], project_root: str):
        self.modules_config = modules_config
        self.project_root = project_root
        
        # Extraer la configuración anidada y los outputs globales
        enabled_outputs = self.modules_config.get('enabled_outputs', {})
        
        image_preparation_config = self.modules_config["modules"]["image_preparation"]
        image_preparation_config['enabled_outputs'] = enabled_outputs

        preprocessing_config = self.modules_config["modules"]['preprocessing']
        preprocessing_config['enabled_outputs'] = enabled_outputs
        
        ocr_config = self.modules_config["modules"]['ocr']
        ocr_config['enabled_outputs'] = enabled_outputs
 
        vectorizing_config = self.modules_config["modules"]['vectorization']
        vectorizing_config['enabled_outputs'] = enabled_outputs

        # Registro de fábricas por nombre de módulo
        self.module_factories: Dict[str, Any] = {
            "image_preparation": ImagePreparationFactory(
                image_preparation_config,
                project_root
            ),
            "preprocessing": PreprocessingFactory(
                preprocessing_config,
                project_root
            ),
            "ocr": OCRFactory(
                ocr_config,
                project_root
            ),
            "vectorization": VectorizingFactory(
                vectorizing_config,
                project_root
            ),
        }

    def get_factory(self, module_name: str):
        """Devuelve la fábrica para el módulo solicitado, o None si no existe."""
        return self.module_factories.get(module_name)

    def get_image_preparation_factory(self) -> ImagePreparationFactory:
        return self.module_factories["image_preparation"]
        # assert isinstance(factory, ImagePreparationFactory)
        # return factory

    def get_preprocessing_factory(self) -> PreprocessingFactory:
        return self.module_factories["preprocessing"]
        # assert isinstance(factory, PreprocessingFactory)
        # return factory
        
    def get_ocr_factory(self) -> OCRFactory:
        return self.module_factories["ocr"] 
        # assert isinstance(factory, OCRFactory) 
        # return factory
        
    def get_vectorizing_factory(self) -> VectorizingFactory:
        return self.module_factories["vectorization"]
        # assert isinstance(factory, VectorizingFactory)
        # return factory
