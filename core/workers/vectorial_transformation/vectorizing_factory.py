# core/workers/factory/vectorizing_factory.py
from typing import Dict, Callable, Any
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.factory.abstract_factory import AbstractBaseFactory
from core.workers.vectorial_transformation.lineal_reconstructor import LinealReconstructor
from core.workers.vectorial_transformation.density_scanner import DensityScanner
from core.workers.vectorial_transformation.geometric_table_structurer import GeometricTableStructurer
#from core.workers.vectorial_transformation. import 


class VectorizingFactory(AbstractBaseFactory[VectorizationAbstractWorker]):
    def create_worker_registry(self) -> Dict[str, Callable[[Dict[str, Any]], VectorizationAbstractWorker]]:
        
        return {
            "lineal": self._create_lineal,
            "dbscan": self._create_scanner,
            "table_structurer": self._create_structurer,
        }
        
    def _create_lineal(self, context: Dict[str, Any]) -> LinealReconstructor:
        return LinealReconstructor(config=self.module_config, project_root=self.project_root)
    
    def _create_scanner(self, context: Dict[str, Any]) -> DensityScanner:
        return DensityScanner(config=self.module_config, project_root=self.project_root)
    
    def _create_structurer(self, context: Dict[str, Any]) -> GeometricTableStructurer:
        return GeometricTableStructurer(config=self.module_config, project_root=self.project_root)
    