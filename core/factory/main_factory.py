# core/workers/factory/main_factory.py
from typing import Dict, Any
from core.workers.image_preparation.image_preparation_factory import ImagePreparationFactory
from core.workers.preprocessing.preprocessing_factory import PreprocessingFactory
from core.workers.ocr.ocr_factory import OCRFactory
from core.workers.vectorial_transformation.vectorizing_factory import VectorizingFactory

class MainFactory:
    """Factory universal que coordina todas las factories de módulos."""

    def __init__(self, manager_config: Dict[str, Any], project_root: str):
        self.manager_config = manager_config
        self.modules_config = self.manager_config.get("modules_config", {})
        self.workers_order = self.manager_config.get("stage_secuence", {})
        self.stage_outputs = self.manager_config.get("enabled_outputs", {})
        self.project_root = project_root
        
        image_preparation_config = self.modules_config["image_preparation"]
        image_preparation_config['enabled_outputs'] = self.stage_outputs["image_load_outputs"]

        preprocessing_config = self.modules_config['preprocessing']
        # preprocessing_config["dpi_range"] = self.modules_config["utils"]
        preprocessing_config['enabled_outputs'] = self.stage_outputs["preprocessing_outputs"]
        
        ocr_config = self.modules_config['ocr']
        ocr_config['enabled_outputs'] = self.stage_outputs["ocr_outputs"]
 
        vectorizing_config = self.modules_config['vectorization']
        vectorizing_config['enabled_outputs'] = self.stage_outputs["vectorization_outputs"]

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
