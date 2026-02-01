# core/workers/preprocessing/preprocessing_factory.py
from typing import Dict, Callable, Any
from core.factory.abstract_worker import  PreprocessingAbstractWorker
from core.factory.abstract_factory import AbstractBaseFactory
from core.workers.preprocessing.restorer import ImageRestorer
from core.workers.preprocessing.sp import DoctorSaltPepper
from core.workers.preprocessing.gauss import GaussianDenoiser
from core.workers.preprocessing.clahe import ClaherEnhancer
from core.workers.preprocessing.sharp import SharpeningEnhancer

class PreprocessingFactory(AbstractBaseFactory[PreprocessingAbstractWorker]):
    def create_worker_registry(self) -> Dict[str, Callable[[Dict[str, Any]], PreprocessingAbstractWorker]]:

        return {
            "restorer": self._create_restorer,
            "sp": self._create_sp,
            "gauss": self._create_gauss,
            "clahe": self._create_clahe,
            "sharp": self._create_sharp,
        }
         
    def _create_restorer(self, context: Dict[str, Any]) -> ImageRestorer:
        return ImageRestorer(config=self.module_config, project_root=self.project_root)

    def _create_sp(self, context: Dict[str, Any]) -> DoctorSaltPepper:
        return DoctorSaltPepper(config=self.module_config, project_root=self.project_root)

    def _create_gauss(self, context: Dict[str, Any]) -> GaussianDenoiser:
        return GaussianDenoiser(config=self.module_config, project_root=self.project_root)

    def _create_clahe(self, context: Dict[str, Any]) -> ClaherEnhancer:
        return ClaherEnhancer(config=self.module_config, project_root=self.project_root)

    def _create_sharp(self, context: Dict[str, Any]) -> SharpeningEnhancer:
        return SharpeningEnhancer(config=self.module_config, project_root=self.project_root)
