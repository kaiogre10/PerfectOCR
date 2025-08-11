# core/workers/factory/main_factory.py
from typing import Dict, Any
from core.workers.image_preparation.image_preparation_factory import ImagePreparationFactory
from core.workers.preprocessing.preprocessing_factory import PreprocessingFactory

class MainFactory:
    """Factory universal que coordina todas las factories de módulos."""

    def __init__(self, modules_config: Dict[str, Any], project_root: str):
        self.modules_config = modules_config
        self.project_root = project_root

        # Registro de fábricas por nombre de módulo
        self.module_factories = {
            "image_loader": ImagePreparationFactory(
                self.modules_config.get('image_loader', {}),
                project_root
            ),
            "preprocessing": PreprocessingFactory(
                self.modules_config.get('preprocessing', {}),
                project_root
            ),
            # agregar más fábricas aquí
        }

    def get_factory(self, module_name: str):
        """Devuelve la fábrica para el módulo solicitado, o None si no existe."""
        return self.module_factories.get(module_name)

    def get_image_preparation_factory(self) -> ImagePreparationFactory:
        factory = self.module_factories["image_loader"]
        assert isinstance(factory, ImagePreparationFactory)
        return factory

    def get_preprocessing_factory(self) -> PreprocessingFactory:
        factory = self.module_factories["preprocessing"]
        assert isinstance(factory, PreprocessingFactory)
        return factory
