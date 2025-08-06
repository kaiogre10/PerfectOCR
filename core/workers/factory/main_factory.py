# core/workers/factory/main_factory.py
from typing import Dict, Any
from core.workers.factory.image_preparation_factory import ImagePreparationFactory
from core.workers.factory.preprocessing_factory import PreprocessingFactory

class MainFactory:
    """Factory universal que coordina todas las factories de módulos."""
    
    def __init__(self, modules_config: Dict[str, Any], project_root: str):
        self.modules_config = modules_config
        self.project_root = project_root
        
        # Crear factories específicas por módulo
        self.module_factories = {
            "image_loader": ImagePreparationFactory(
                self.modules_config.get('image_loader', {}), 
                project_root
            ),
            "preprocessing": PreprocessingFactory(
                self.modules_config.get('preprocessing', {}), 
                project_root
            ),
        }
    
    def get_image_preparation_factory(self) -> ImagePreparationFactory:
        return self.module_factories["image_loader"]
    
    def get_preprocessing_factory(self) -> PreprocessingFactory:
        return self.module_factories["preprocessing"]