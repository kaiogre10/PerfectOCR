# core/domain/secret_job.py
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class FragmentedBoundingBox:
    """Bounding box inmutable con validación"""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    
    def __post_init__(self):
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("Bounding box inválido")
    
    @property
    def width(self) -> float:
        return self.x_max - self.x_min
    
    @property
    def height(self) -> float:
        return self.y_max - self.y_min
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    @property
    def centroid(self) -> Tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)

@dataclass
class FragmentedPolygonGeometry:
    """Geometría de polígono con métodos útiles"""
    polygon_coords: List[Tuple[float, float]]
    bounding_box: FragmentedBoundingBox
    centroid: Tuple[float, float]
    
    def __post_init__(self):
        if len(self.polygon_coords) < 3:
            raise ValueError("Polígono debe tener al menos 3 puntos")
    
    @property
    def perimeter(self) -> float:
        """Calcula perímetro del polígono"""
        perimeter = 0.0
        for i in range(len(self.polygon_coords)):
            p1 = self.polygon_coords[i]
            p2 = self.polygon_coords[(i + 1) % len(self.polygon_coords)]
            perimeter += np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        return perimeter
    
@dataclass
class Polygon:
    """Polígono con validación y métodos útiles"""
    polygon_id: str
    geometry: FragmentedPolygonGeometry
    line_id: str
    cropped_img: Optional[np.ndarray] = None
    padding_coords: Optional[List[float]] = None
    was_fragmented: bool = False
    text: Optional[str] = None
    confidence: Optional[float] = None
    
    def __post_init__(self):
        if not self.polygon_id.startswith("poly_"):
            raise ValueError("ID de polígono debe empezar con 'poly_'")
        if not self.line_id.startswith("line_"):
            raise ValueError("ID de línea debe empezar con 'line_'")
        if self.confidence is not None and not (0 <= self.confidence <= 100):
            raise ValueError("Confianza debe estar entre 0 y 100")
    
    @property
    def has_text(self) -> bool:
        return self.text is not None and len(self.text.strip()) > 0
    
    @property
    def is_high_confidence(self) -> bool:
        return self.confidence is not None and self.confidence >= 80.0

@dataclass
class LineInfo:
    """Información de línea con métodos útiles"""
    line_id: str
    bounding_box: FragmentedBoundingBox
    centroid: Tuple[float, float]
    new_polygon_ids: List[str]
