# core/workers/ocr/ocr_factory.py
from typing import Dict, Callable, Any, Optional
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
    def __init__(self, module_config: Dict[str, Any], project_root: str):
        super().__init__(module_config, project_root)
        self._shared_refiner_workers: Optional[Dict[str, OCRAbstractWorker]] = None
    
    @property
    def shared_refiner_workers(self) -> Dict[str, OCRAbstractWorker]:
        """Crea workers compartidos del refinador UNA SOLA VEZ."""
        if self._shared_refiner_workers is None:
            self._shared_refiner_workers = {
                "clasificator": SemanticClasificator(config=self.module_config, project_root=self.project_root),
                "cleaner": TextCleaner(config=self.module_config, project_root=self.project_root),
                "fragmenter": Fragmenter(config=self.module_config, project_root=self.project_root),
                "corrector": TextCorrector(config=self.module_config, project_root=self.project_root)
            }
        return self._shared_refiner_workers
    
    def create_worker_registry(self) -> Dict[str, Callable[[Dict[str, Any]], OCRAbstractWorker]]:
        return {
            "paddle_wrapper": self._create_paddle_wrapper,
            "text_refiner": self._create_refiner,
            "data_finder": self._create_finder,
        }
        
    def _create_paddle_wrapper(self, context: Dict[str, Any]) -> PaddleOCRWrapper:
        return PaddleOCRWrapper(config=self.module_config, project_root=self.project_root) 
    
    def _create_refiner(self, context: Dict[str, Any]) -> Refiner:
        workers = self.shared_refiner_workers
        return Refiner(
            config=self.module_config, 
            project_root=self.project_root,
            clasificator=workers["clasificator"], #type: ignore
            cleaner=workers["cleaner"], #type: ignore
            fragmenter=workers["fragmenter"], #type: ignore
            corrector=workers["corrector"] #type: ignore
        ) 
    
    def _create_finder(self, context: Dict[str, Any]) -> DataFinder:
        return DataFinder(config=self.module_config, project_root=self.project_root)