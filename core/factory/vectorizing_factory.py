# core/workers/factory/vectorizing_factory.py
from typing import Dict, Callable, Any
from domain.abstract_worker import VectorizationAbstractWorker
from domain.abstract_factory import AbstractBaseFactory
from core.workers.vectorial_transformation.lineal_reconstructor import LinealReconstructor
from core.workers.vectorial_transformation.matricial_cosine import MatricialCusine
from core.workers.vectorial_transformation.geometric_table_structurer import GeometricTableStructurer
from core.workers.vectorial_transformation.math_max import MatrixSolver
from core.workers.vectorial_transformation.data_collector import FinalStructurer

class VectorizingFactory(AbstractBaseFactory[VectorizationAbstractWorker]):
    def create_registry(self) -> Dict[str, Callable[[Dict[str, Any]], VectorizationAbstractWorker]]:
        
        return {
            "lineal": self._create_lineal,
            "cos_sim": self._create_cosmatrix,
            "table_structurer": self._create_table_structurer,
            "math_max": self._create_mathmax,
            "collector": self._create_structurer,
        }

    def _create_lineal(self, context: Dict[str, Any]) -> LinealReconstructor:
        return LinealReconstructor(config=self.module_config, project_root=self.project_root)
    
    def _create_cosmatrix(self, context: Dict[str, Any]) -> MatricialCusine:
        return MatricialCusine(config=self.module_config, project_root=self.project_root)

    def _create_table_structurer(self, context: Dict[str, Any]) -> GeometricTableStructurer:
        return GeometricTableStructurer(config=self.module_config, project_root=self.project_root)
        
    def _create_mathmax(self, context: Dict[str, Any]) -> MatrixSolver:
        return MatrixSolver(config=self.module_config, project_root=self.project_root)

    def _create_structurer(self, context: Dict[str, Any]) -> FinalStructurer:
        return FinalStructurer(config=self.module_config, project_root=self.project_root)