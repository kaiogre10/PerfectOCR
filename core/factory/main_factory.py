# core/factory/main_factory.py
from typing import Dict, Any, List, Tuple, Optional, Type

from core.factory.abstract_factory import AbstractBaseFactory
from core.factory.abstract_stager import AbstractStager
from core.workers.image_preparation.image_preparation_factory import ImagePreparationFactory
from core.workers.preprocessing.preprocessing_factory import PreprocessingFactory
from core.workers.ocr.ocr_factory import OCRFactory
from core.workers.vectorial_transformation.vectorizing_factory import VectorizingFactory
from core.pipeline.image_preparation_stager import ImagePreparationStager
from core.pipeline.preprocessing_stager import PreprocessingStager
from core.pipeline.ocr_stager import OCRStager
from core.pipeline.vectorization_stager import VectorizationStager

PipelineStagerEntry = Tuple[
    Type[AbstractBaseFactory[Any]],
    Type[AbstractStager],
]

# Mismas claves que manager_config (ConfigService) y entradas de pipeline_secuence con stager.
PIPELINE_STAGER_REGISTRY: Dict[str, PipelineStagerEntry] = {
    "image_preparation_stager": (ImagePreparationFactory, ImagePreparationStager),
    "preprocessing_stager": (PreprocessingFactory, PreprocessingStager),
    "ocr_stager": (OCRFactory, OCRStager),
    "vectorization_stager": (VectorizingFactory, VectorizationStager),
}


class MainFactory:
    """Ensambla stagers del pipeline a partir de aviable_stagers y manager_config."""

    def __init__(self, manager_config: Dict[str, Any], project_root: str):
        self.manager_config = manager_config
        self.project_root = project_root
        self.stagers_registry = PIPELINE_STAGER_REGISTRY

    def get_all_stagers(
        self,
        aviable_stagers: List[Tuple[str, List[str]]],
        context: Optional[Dict[str, Any]] = None,
    ) -> List[AbstractStager]:
        stagers: List[AbstractStager] = []

        for (stager, workers) in aviable_stagers:
            if not workers:
                continue

            pipeline_entry = self.stagers_registry.get(stager)
            if pipeline_entry is None:
                continue

            config_stage: Any = self.manager_config.get(stager)
            if not config_stage:
                continue

            factory_cls, stager_cls = pipeline_entry
            factory = factory_cls(config_stage, self.project_root)
            workers_factory = factory.create_components(workers, context)
            stagers.append(
                stager_cls(workers_factory, config_stage, self.project_root)
            )

        return stagers
