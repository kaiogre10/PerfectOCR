# core/pipeline/stagers_factory.py
from core.workers.image_preparation.image_preparation_factory import ImagePreparationFactory
from core.workers.preprocessing.preprocessing_factory import PreprocessingFactory
from core.workers.ocr.ocr_factory import OCRFactory
from core.workers.vectorial_transformation.vectorizing_factory import VectorizingFactory
from core.pipeline.image_preparation_stager import ImagePreparationStager
from core.pipeline.preprocessing_stager import PreprocessingStager
from core.pipeline.ocr_stager import OCRStager
from core.pipeline.vectorization_stager import VectorizationStager
from typing import Dict, Any, Tuple, List
import logging

keyimglo ="image_preparation_stager"
keypre = "preprocessing_stager"
ocrkey = "ocr_stager"
veckey = "vectorization_stager"

list_keys: List[str] = list([keyimglo, keypre, ocrkey, veckey])

logger = logging.getLogger(__name__)

class StagersFactory:
    """Crea workers y ensambla stagers de forma uniforme."""
    def __init__(self, modules_config: Dict[str, Dict[str, Any]], project_root: str, stagging: List[Tuple[str, List[str]]]):
        self.project_root = project_root
        self.stagging = stagging
        self.modules_config = modules_config
        
        factories_dict: Dict[str, Any] = {
            keyimglo: self.get_image_preparation_factory(self.modules_config.get(keyimglo)), # type: ignore
            keypre: self.get_preprocessing_factory(self.modules_config.get(keypre)), # type: ignore
            ocrkey: self.get_ocr_factory(self.modules_config.get(ocrkey)), # type: ignore
            veckey: self.get_vectorizing_factory(self.modules_config.get(veckey)) # type: ignore
        }

        stagers_dict: Dict[str, Any] = {
            keyimglo: ImagePreparationStager,
            keypre: PreprocessingStager,
            ocrkey: OCRStager,
            veckey: VectorizationStager,    
        }
        all_stagers: List[Any] = self.buil_stagers(factories_dict, stagers_dict)
        self.all_stagers = all_stagers

    def buil_stagers(self, factories_dict: Dict[str, Any], stagers_dict: Dict[str, Any]) -> List[Any]:
        # logger.info(f"Stagers: {stagging}")
        stagers: List[Any] = []
        
        for (stage, workers) in self.stagging:
            stage_config  = self.modules_config.get(stage) # Configuración por etapa
            if not workers or not stage:
                # logger.info(f"STAGE SIN WORKERS: '{stage}'")
                continue
            try:
                workers_order = stage_config.get(stage) # Pipeline_config
                # logger.info(f"STAGE: {stage}: WORKERS: {workers_order}")

                config = self.modules_config.get(stage)
                # logger.info(f"STAGE DEBUGG: {stage}")
                factory = factories_dict[stage]
                workers_created = factory.create_components(workers_order)
                stager = stagers_dict.pop(stage)
                stagers.append(stager(workers_created, config, self.project_root))
            except AttributeError as e:
                logger.info(f"error stagggin: {e}", exc_info=True)
        return stagers

    def get_all_stagers(self) -> List[Any]:
        return self.all_stagers

    def get_image_preparation_factory(self, config: Dict[str, Any]) -> ImagePreparationFactory:
        return ImagePreparationFactory(config, self.project_root)

    def get_preprocessing_factory(self, config: Dict[str, Any]) -> PreprocessingFactory:
        return PreprocessingFactory(config, self.project_root)

    def get_ocr_factory(self, config: Dict[str, Any]) -> OCRFactory:
        return OCRFactory(config, self.project_root)

    def get_vectorizing_factory(self, config: Dict[str, Any]) -> VectorizingFactory:
        return VectorizingFactory(config, self.project_root)