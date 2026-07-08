# core/pipeline/stagers_factory.py
from core.factory.image_preparation_factory import ImagePreparationFactory
from core.factory.preprocessing_factory import PreprocessingFactory
from core.factory.ocr_factory import OCRFactory
from core.factory.vectorizing_factory import VectorizingFactory
from core.pipeline.image_preparation_stager import ImagePreparationStager
from core.pipeline.preprocessing_stager import PreprocessingStager
from core.pipeline.ocr_stager import OCRStager
from core.pipeline.vectorization_stager import VectorizationStager
from typing import Dict, Any, Tuple, List, Optional

keyimglo ="image_preparation_stager"
keypre = "preprocessing_stager"
ocrkey = "ocr_stager"
veckey = "vectorization_stager"

class MainFactory:
    """Crea workers y ensambla stagers de forma uniforme."""
    __slots__ = ("project_root", "stagging", "modules_config", "all_stagers")
    def __init__(self, modules_config: Dict[str, Tuple[Dict[str, Any], List[str]]], project_root: str, stagging: List[Tuple[str, Optional[List[str]]]]):
        self.project_root = project_root
        self.stagging = stagging
        self.modules_config = modules_config
        
        factories_dict: Dict[str, Any] = {
            keyimglo: self.get_image_preparation_factory,
            keypre: self.get_preprocessing_factory,
            ocrkey: self.get_ocr_factory,
            veckey: self.get_vectorizing_factory,
        }

        stagers_dict: Dict[str, Any] = {
            keyimglo: ImagePreparationStager,
            keypre: PreprocessingStager,
            ocrkey: OCRStager,
            veckey: VectorizationStager,    
        }
        all_stagers: List[Any] = self.build_stagers(factories_dict, stagers_dict)
        del factories_dict
        self.all_stagers = all_stagers

    def build_stagers(self, factories_dict: Dict[str, Any], stagers_dict: Dict[str, Any]) -> List[Any]:
        stagers: List[Any] = []
        for (stage, workers) in self.stagging:
            stage_config = self.modules_config.get(stage) # Configuración por etapa
            if workers is None or not workers or not stage or not stage_config:
                continue
            try:
                workers_order: List[str] = stage_config[1] # Pipeline_config 
                if not workers_order:
                    continue
                config = self.modules_config.get(stage)
                if config is None:
                    continue
                factory = factories_dict[stage](config[0])
                if factory is None or not workers_order:
                    continue
                workers_created = factory.create_components(workers_order)
                stager = stagers_dict.pop(stage)
                stagers.append(stager(workers_created, config, self.project_root))
                continue
            except AttributeError:
                raise
        return stagers

    def get_all_stagers(self) -> List[Any]:
        return self.all_stagers

    def get_image_preparation_factory(self, config: Dict[str, Any]) -> ImagePreparationFactory:
        return ImagePreparationFactory(config, self.project_root)

    def get_preprocessing_factory(self, config: Dict[str, Any]) -> PreprocessingFactory:
        return PreprocessingFactory(config, self.project_root)

    def get_ocr_factory(self, config: Dict[str, Any]) -> OCRFactory:
        return OCRFactory(config, self.project_root) # type: ignore

    def get_vectorizing_factory(self, config: Dict[str, Any]) -> VectorizingFactory:
        return VectorizingFactory(config, self.project_root)