# core/workers/factory/preprocessing_factory.py
from typing import Dict, Callable, Any
from core.factory.abstract_worker import  PreprossesingAbstractWorker
from core.factory.abstract_factory import AbstractBaseFactory
from core.workers.preprocessing.moire import MoireDenoiser
from core.workers.preprocessing.sp import DoctorSaltPepper
from core.workers.preprocessing.gauss import GaussianDenoiser
from core.workers.preprocessing.clahe import ClaherEnhancer
from core.workers.preprocessing.sharp import SharpeningEnhancer
# from core.workers.preprocessing.binarization import Binarizator
# from core.workers.preprocessing.fragmentator import PolygonFragmentator

class PreprocessingFactory(AbstractBaseFactory[PreprossesingAbstractWorker]):
    def create_worker_registry(self) -> Dict[str, Callable[[Dict[str, Any]], PreprossesingAbstractWorker]]:

        return {
            "moire": self._create_moire,
            "sp": self._create_sp,
            "gauss": self._create_gauss,
            "clahe": self._create_clahe,
            "sharp": self._create_sharp,
            # "binarization": self._create_binarization,
            # "fragmentator": self._create_fragmentator,
        }
    # 
    def _create_moire(self, context: Dict[str, Any]) -> MoireDenoiser:
        moire_config = self.module_config.get('moire', {})
        return MoireDenoiser(config=moire_config, project_root=self.project_root)

    def _create_sp(self, context: Dict[str, Any]) -> DoctorSaltPepper:
        sp_config = self.module_config.get('median_filter', {})
        return DoctorSaltPepper(config=sp_config, project_root=self.project_root)

    def _create_gauss(self, context: Dict[str, Any]) -> GaussianDenoiser:
        gauss_config = self.module_config.get('bilateral_params', {})
        return GaussianDenoiser(config=gauss_config, project_root=self.project_root)

    def _create_clahe(self, context: Dict[str, Any]) -> ClaherEnhancer:
        clahe_config = self.module_config.get('clahe', {})
        return ClaherEnhancer(config=clahe_config, project_root=self.project_root)

    def _create_sharp(self, context: Dict[str, Any]) -> SharpeningEnhancer:
        sharp_config = self.module_config.get('sharp', {})
        return SharpeningEnhancer(config=sharp_config, project_root=self.project_root)

    # def _create_binarization(self, context: Dict[str, Any]) -> Binarizator:
    #     binarization_config = self.module_config.get('binarize', {})
    #     return Binarizator(config=binarization_config, project_root=self.project_root)    
        
    # def _create_fragmentator(self, context: Dict[str, Any]) -> PolygonFragmentator:
    #    fragment_config = self.module_config.get('fragmentation', {})
    #    return PolygonFragmentator(config=fragment_config, project_root=self.project_root)