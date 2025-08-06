# core/workers/factory/preprocessing_factory.py
from typing import Dict, Callable, Any
from core.workers.factory.abstract_factory import AbstractBaseFactory
from core.workers.preprocessing.binarization import Binarizator
from core.workers.preprocessing.clahe import ClaherEnhancer
from core.workers.preprocessing.sharp import SharpeningEnhancer

class PreprocessingFactory(BaseModuleFactory):
    """Factory para workers de preprocesamiento."""
    
    def _create_worker_registry(self) -> Dict[str, Callable]:
        return {
            "binarization": self._create_binarization,
            "clahe": self._create_clahe,
            "sharp": self._create_sharp
        }
    
    def _create_binarization(self, context: Dict[str, Any]) -> Binarizator:
        binarization_config = self.module_config.get('binarization', {})
        return Binarization(config=binarization_config, project_root=self.project_root)
    
    def _create_clahe(self, context: Dict[str, Any]) -> ClaherEnhancer:
        clahe_config = self.module_config.get('clahe', {})
        return CLAHE(config=clahe_config, project_root=self.project_root)
    
    def _create_sharp(self, context: Dict[str, Any]) -> SharpeningEnhancer:
        sharp_config = self.module_config.get('sharp', {})
        return Sharp(config=sharp_config, project_root=self.project_root)