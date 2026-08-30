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
from domain.class_models import StageKeys

class MainFactory:
    """Crea workers y ensambla stagers de forma uniforme."""
    def __init__(self, project_root: str, modules_config: Dict[str, Tuple[Dict[str, Any], List[str]]]):
        self.project_root = project_root
        self.modules_config = modules_config
        
    def get_all_stagers(self, stagging: List[Tuple[str, Optional[List[str]]]]) -> List[Any]:
        stagers: List[Any] = []
        stagers_dict, factories_dict = self.get_dicts()
        for (stage, workers) in stagging:
            stage_config = self.modules_config.get(stage) # Configuración por etapa
            if workers is None or not workers or not stage or not stage_config:
                continue
            
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
            factory.registry.clear()
            stager = stagers_dict.pop(stage)
            stagers.append(stager(workers_created, config, self.project_root))
            continue
            
        return stagers
    
    def get_dicts(self) -> Tuple[Dict[str, Any],  Dict[str, Any]]:
        """"StagersDict, FactoriesDict"""
        factories_dict: Dict[str, Any] = {
            StageKeys.IMGPREP_KEY: self.get_image_preparation_factory,
            StageKeys.PREPRO_KEY: self.get_preprocessing_factory,
            StageKeys.OCR_KEY: self.get_ocr_factory,
            StageKeys.VECT_KEY: self.get_vectorizing_factory,
        }

        stagers_dict: Dict[str, Any] = {
            StageKeys.IMGPREP_KEY: ImagePreparationStager,
            StageKeys.PREPRO_KEY: PreprocessingStager,
            StageKeys.OCR_KEY: OCRStager,
            StageKeys.VECT_KEY: VectorizationStager,
        }
        return stagers_dict, factories_dict

    def get_image_preparation_factory(self, config: Dict[str, Any]) -> ImagePreparationFactory:
        return ImagePreparationFactory(config, self.project_root)

    def get_preprocessing_factory(self, config: Dict[str, Any]) -> PreprocessingFactory:
        return PreprocessingFactory(config, self.project_root)

    def get_ocr_factory(self, config: Dict[str, Any]) -> OCRFactory:
        return OCRFactory(config, self.project_root)

    def get_vectorizing_factory(self, config: Dict[str, Any]) -> VectorizingFactory:
        return VectorizingFactory(config, self.project_root)
