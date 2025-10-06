# core/workers/ocr/ocr_factory.py
from typing import Dict, Callable, Any
from core.factory.abstract_worker import OCRAbstractWorker
from core.factory.abstract_factory import AbstractBaseFactory
from core.workers.ocr.paddle_wrapper import PaddleOCRWrapper
from core.workers.ocr.text_refiner import Refiner
# from core.workers.ocr.text_cleaner import TextCleaner
# from core.workers.ocr.semantic_clasificator import SemanticClasificator
# from core.workers.ocr.fragmenter import Fragmenter
from core.workers.vectorial_transformation.data_finder import DataFinder

class OCRFactory(AbstractBaseFactory[OCRAbstractWorker]):
    def create_worker_registry(self) -> Dict[str, Callable[[Dict[str, Any]], OCRAbstractWorker]]:
        
        return {
            "paddle_wrapper": self._create_paddle_wrapper,
            "text_refiner": self._create_refiner,
            # "text_cleaner": self._create_text_cleaner,
            # "semantic_clasificator": self._create_clasificator,
            # "fragmenter": self._create_fragmenter,
            "data_finder": self._create_finder,

        }
        
    def _create_paddle_wrapper(self, context: Dict[str, Any]) -> PaddleOCRWrapper:
        return PaddleOCRWrapper(config=self.module_config, project_root=self.project_root) 
    
    def _create_refiner(self, context: Dict[str, Any]) -> Refiner:
        return Refiner(config=self.module_config, project_root=self.project_root) 

    # def _create_text_cleaner(self, context: Dict[str, Any]) -> TextCleaner:
    #     return TextCleaner(config=self.module_config, project_root=self.project_root)
        
    # def _create_clasificator(self, context: Dict[str, Any]) -> SemanticClasificator:
    #     return SemanticClasificator(config=self.module_config, project_root=self.project_root)

    # def _create_fragmenter(self, context: Dict[str, Any]) -> Fragmenter:
    #     return Fragmenter(config=self.module_config, project_root=self.project_root)
    
    def _create_finder(self, context: Dict[str, Any]) -> DataFinder:
        data_finder_config = context.get("data_finder_config", {})
        return DataFinder(config=data_finder_config, cfg=self.module_config, project_root=self.project_root)