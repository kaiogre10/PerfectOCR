# core/workers/factory/image_preparation_factory.py
from typing import Dict, Callable, Any
from core.workers.factory.abstract_factory import AbstractBaseFactory
from core.workers.image_preparation.cleanner import ImageCleaner
from core.workers.image_preparation.angle_corrector import AngleCorrector
from core.workers.image_preparation.geometry_detector import GeometryDetector
from core.workers.image_preparation.lineal_reconstructor import LineReconstructor
from core.workers.image_preparation.poly_gone import PolygonExtractor

class ImagePreparationFactory(AbstractBaseFactory):
    """Factory para workers de preparación de imagen."""
    
    def _create_worker_registry(self) -> Dict[str, Callable]:
        return {
            "cleaner": self._create_cleaner,
            "angle_corrector": self._create_angle_corrector,
            "geometry_detector": self._create_geometry_detector,
            "line_reconstructor": self._create_line_reconstructor,
            "polygon_extractor": self._create_polygon_extractor
        }
    
    def _create_cleaner(self, context: Dict[str, Any]) -> ImageCleaner:
        cleaning_config = self.module_config.get('cleaning', {})
        return ImageCleaner(config=cleaning_config, project_root=self.project_root)
    
    def _create_angle_corrector(self, context: Dict[str, Any]) -> AngleCorrector:
        deskew_config = self.module_config.get('deskew', {})
        return AngleCorrector(config=deskew_config, project_root=self.project_root)
    
    def _create_geometry_detector(self, context: Dict[str, Any]) -> GeometryDetector:
        paddle_config = self.module_config.get('paddle_config', {})
        return GeometryDetector(paddle_config=paddle_config, project_root=self.project_root)
    
    def _create_line_reconstructor(self, context: Dict[str, Any]) -> LineReconstructor:
        return LineReconstructor(config={}, project_root=self.project_root)
    
    def _create_polygon_extractor(self, context: Dict[str, Any]) -> PolygonExtractor:
        cutting_config = self.module_config.get('cutting', {})
        return PolygonExtractor(config=cutting_config, project_root=self.project_root)