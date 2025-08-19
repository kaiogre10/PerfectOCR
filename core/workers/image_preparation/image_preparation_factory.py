# core/workers/image_preparation_factory.py
from typing import Dict, Callable, Any
from core.factory.abstract_worker import ImagePrepAbstractWorker
from core.factory.abstract_factory import AbstractBaseFactory
from core.workers.image_preparation.cleanner import ImageCleaner
from core.workers.image_preparation.angle_corrector import AngleCorrector
from core.workers.image_preparation.geometry_detector import GeometryDetector
from core.workers.image_preparation.poly_gone import PolygonExtractor

class ImagePreparationFactory(AbstractBaseFactory[ImagePrepAbstractWorker]):    
    def create_worker_registry(self) -> Dict[str, Callable[[Dict[str, Any]], ImagePrepAbstractWorker]]:
        
        return {
            "cleaner": self._create_cleaner,
            "angle_corrector": self._create_angle_corrector,
            "geometry_detector": self._create_geometry_detector,
            "polygon_extractor": self._create_polygon_extractor
        }
    
    def _create_cleaner(self, context: Dict[str, Any]) -> ImageCleaner:
        cleaning_config = self.module_config.get('cleaning', {})
        return ImageCleaner(config=cleaning_config, project_root=self.project_root)
    
    def _create_angle_corrector(self, context: Dict[str, Any]) -> AngleCorrector:
        deskew_config = self.module_config.get('deskew', {})
        return AngleCorrector(config=deskew_config, project_root=self.project_root)
    
    def _create_geometry_detector(self, context: Dict[str, Any]) -> GeometryDetector:
        paddleocr = context.get('paddle_det_config', {})
        return GeometryDetector(config=paddleocr, project_root=self.project_root)
    
    def _create_polygon_extractor(self, context: Dict[str, Any]) -> PolygonExtractor:
        cutting_config = self.module_config.get('cutting', {})
        return PolygonExtractor(config=cutting_config, project_root=self.project_root)