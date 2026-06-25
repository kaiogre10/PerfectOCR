# core/workers/image_preparation_factory.py
from typing import Dict, Callable, Any
from domain.abstract_worker import ImagePrepAbstractWorker
from domain.abstract_factory import AbstractBaseFactory
from core.workers.image_preparation.image_loader import ImageLoader
from core.workers.image_preparation.ink_enhancer import InkCorrector
from core.workers.image_preparation.angle_corrector import AngleCorrector
from core.workers.image_preparation.geometry_detector import GeometryDetector
from core.workers.image_preparation.poly_gone import PolygonExtractor

class ImagePreparationFactory(AbstractBaseFactory[ImagePrepAbstractWorker]):
    def create_registry(self) -> Dict[str, Callable[[Dict[str, Any]], ImagePrepAbstractWorker]]:
        
        return {
            'image_loader': self._create_loader,
            "angle_corrector": self._create_angle_corrector,
            "ink_enhancement": self._create_inker,
            "geometry_detector": self._create_geometry_detector,
            "polygon_extractor": self._create_polygon_extractor
        }
    
    def _create_loader(self, context: Dict[str, Any]) -> ImageLoader:
        return ImageLoader(config=self.module_config, project_root=self.project_root)
    
    def _create_angle_corrector(self, context: Dict[str, Any]) -> AngleCorrector:
        return AngleCorrector(config=self.module_config, project_root=self.project_root)

    def _create_inker(self, context: Dict[str, Any]) -> InkCorrector:
        return InkCorrector(config=self.module_config, project_root=self.project_root)
    
    def _create_geometry_detector(self, context: Dict[str, Any]) -> GeometryDetector:
        return GeometryDetector(config=self.module_config, project_root=self.project_root)
    
    def _create_polygon_extractor(self, context: Dict[str, Any]) -> PolygonExtractor:
        return PolygonExtractor(config=self.module_config, project_root=self.project_root)