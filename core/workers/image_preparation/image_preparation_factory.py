# core/workers/image_preparation_factory.py
import time
t_import0 = time.perf_counter()

from typing import Dict, Callable, Any
t_import1 = time.perf_counter()
from core.factory.abstract_worker import ImagePrepAbstractWorker
print(f"Desde IMAGEPREPARATIONFACTORY: Import ImagePrepAbstractWorker en {time.perf_counter() - t_import1:.6f}s")

t_import2 = time.perf_counter()
from core.factory.abstract_factory import AbstractBaseFactory
print(f"Desde IMAGEPREPARATIONFACTORY: Import AbstractBaseFactory en {time.perf_counter() - t_import2:.6f}s")

t_import3 = time.perf_counter()
from core.workers.image_preparation.cleanner import ImageCleaner
print(f"Desde IMAGEPREPARATIONFACTORY: Import ImageCleaner en {time.perf_counter() - t_import3:.6f}s")

t_import4 = time.perf_counter()
from core.workers.image_preparation.angle_corrector import AngleCorrector
print(f"Desde IMAGEPREPARATIONFACTORY: Import AngleCorrector en {time.perf_counter() - t_import4:.6f}s")

t_import5 = time.perf_counter()
from core.workers.image_preparation.geometry_detector import GeometryDetector
print(f"Desde IMAGEPREPARATIONFACTORY: Import GeometryDetector en {time.perf_counter() - t_import5:.6f}s")

t_import6 = time.perf_counter()
from core.workers.image_preparation.poly_gone import PolygonExtractor
print(f"Desde IMAGEPREPARATIONFACTORY: Import PolygonExtractor en {time.perf_counter() - t_import6:.6f}s")
print(f"Desde IMAGEPREPARATIONFACTORY: Tiempo total de importaciones en {time.perf_counter() - t_import0:.6f}s")

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