# core/workers/ocr/ocr_factory.py
from typing import Dict, Callable, Any, List
from core.factory.abstract_worker import OCRAbstractWorker
from core.factory.abstract_factory import AbstractBaseFactory
from core.workers.ocr.paddle_wrapper import PaddleOCRWrapper
from core.workers.ocr.text_refiner import Refiner
from core.workers.ocr.text_cleaner import TextCleaner
from core.workers.ocr.semantic_clasificator import SemanticClasificator
from core.workers.ocr.fragmenter import Fragmenter
from core.workers.ocr.text_corrector import TextCorrector
from core.workers.ocr.data_finder import DataFinder

class OCRFactory(AbstractBaseFactory[OCRAbstractWorker]):
    def create_worker_registry(self) -> Dict[str, Callable[[Dict[str, Any]], OCRAbstractWorker]]:
        
        return {
            "paddle_wrapper": self._create_paddle_wrapper,
            "text_refiner": self._create_refiner,
            "data_finder": self._create_finder,
        }
        
    def _create_paddle_wrapper(self, context: Dict[str, Any]) -> PaddleOCRWrapper:
        return PaddleOCRWrapper(config=self.module_config, project_root=self.project_root) 
    
    def _create_refiner(self, context: Dict[str, Any]) -> Refiner:
        # 1. Crear las instancias que el Refiner necesita
        clasificator = SemanticClasificator(config=self.module_config, project_root=self.project_root)
        cleaner = TextCleaner(config=self.module_config, project_root=self.project_root)
        fragmenter = Fragmenter(config=self.module_config, project_root=self.project_root)
        corrector = TextCorrector(config=self.module_config, project_root=self.project_root)

        # 2. Inyectar las instancias en el constructor del Refiner
        return Refiner(
            config=self.module_config, 
            project_root=self.project_root,
            clasificator=clasificator,
            cleaner=cleaner,
            fragmenter=fragmenter,
            corrector=corrector
        ) 
    
    def _create_finder(self, context: Dict[str, Any]) -> DataFinder:
        self.noise_words: List[str] = context["noise_words"]
        self.config = {
            'noise_words': self.noise_words,
            "config": self.module_config
        }
        return DataFinder(config=self.config, project_root=self.project_root)