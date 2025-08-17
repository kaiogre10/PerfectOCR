# core/workers/factory/main_factory.py
import time
t_import0 = time.perf_counter()

from typing import Dict, Any

t_import1 = time.perf_counter()
from core.workers.image_preparation.image_preparation_factory import ImagePreparationFactory
print(f"Desde MAIN_FACTORY: Import IMAGEPREPARATIONFACTORY en {time.perf_counter() - t_import1:.6f}s")

t_import2 = time.perf_counter()
from core.workers.preprocessing.preprocessing_factory import PreprocessingFactory
print(f"Desde MAIN_FACTORY: Import ProcessingFactory en {time.perf_counter() - t_import2:.6f}s")

from core.workers.ocr.ocr_factory import OCRFactory

t_import3 = time.perf_counter()
from core.workers.vectorial_transformation.vectorizing_factory import VectorizingFactory
print(f"Desde MAIN_FACTORY: Import VectorFactory en {time.perf_counter() - t_import3:.6f}s")
print(f"Desde MAIN_FACTORY: Tiempo total de importaciones en {time.perf_counter() - t_import0:.6f}s")

class MainFactory:
    """Factory universal que coordina todas las factories de módulos."""

    def __init__(self, modules_config: Dict[str, Any], project_root: str):
        self.modules_config = modules_config
        self.project_root = project_root

        # Extraer la configuración anidada y los outputs globales
        nested_modules = self.modules_config.get('modules', {})
        enabled_outputs = self.modules_config.get('enabled_outputs', {})

        # Crear configuración enriquecida para preprocesamiento
        preprocessing_config = nested_modules.get('preprocessing', {}).copy()
        preprocessing_config['enabled_outputs'] = enabled_outputs
        
        ocr_config = nested_modules.get('ocr', {}).copy() # <-- Añadido
        ocr_config['enabled_outputs'] = enabled_outputs # <-- Añadido
 
        
        vectorizing_config = nested_modules.get('vectorization', {}).copy()
        vectorizing_config['enabled_outputs'] = enabled_outputs

        # Registro de fábricas por nombre de módulo
        self.module_factories = {
            "image_loader": ImagePreparationFactory(
                nested_modules.get('image_loader', {}),
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
        factory = self.module_factories["image_loader"]
        assert isinstance(factory, ImagePreparationFactory)
        return factory

    def get_preprocessing_factory(self) -> PreprocessingFactory:
        factory = self.module_factories["preprocessing"]
        assert isinstance(factory, PreprocessingFactory)
        return factory
        
    def get_ocr_factory(self) -> OCRFactory:
        factory = self.module_factories["ocr"] # <-- Corregido
        assert isinstance(factory, OCRFactory) # <-- Corregido
        return factory

        
    def get_vectorizing_factory(self) -> VectorizingFactory:
        factory = self.module_factories["vectorization"]
        assert isinstance(factory, VectorizingFactory)
        return factory
