# core/workers/ocr/ocr_factory.py
from typing import Dict, Callable, Any
from core.factory.abstract_worker import OCRAbstractWorker
from core.factory.abstract_factory import AbstractBaseFactory
from core.workers.ocr.binarization import Binarizator
from core.workers.ocr.paddle_wrapper import PaddleOCRWrapper
from core.workers.ocr.text_cleaner import TextCleaner

class OCRFactory(AbstractBaseFactory[OCRAbstractWorker]):
    def create_worker_registry(self) -> Dict[str, Callable[[Dict[str, Any]], OCRAbstractWorker]]:
        
        return {
            "binarizator": self._create_binarizator,
            "paddle_wrapper": self._create_paddle_wrapper,
            "text_cleaner": self._create_text_cleaner,
        }
    
    def _create_binarizator(self, context: Dict[str, Any]) -> Binarizator:
        return Binarizator(config=self.module_config, project_root=self.project_root)
        
    def _create_paddle_wrapper(self, context: Dict[str, Any]) -> PaddleOCRWrapper:
        # ESPERA recibir la config de paddle en el context
        paddle_rec_config = context.get('paddle_rec_config', {})
        return PaddleOCRWrapper(config=paddle_rec_config, project_root=self.project_root)
        
    def _create_text_cleaner(self, context: Dict[str, Any]) -> TextCleaner:
        return TextCleaner(config=self.module_config, project_root=self.project_root)