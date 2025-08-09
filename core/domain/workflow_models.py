# core/domain/workflow_models.py
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

@dataclass
class PaddingGeometry:
    padding_bbox: List[float]
    padd_centroid: List[float]
    padding_coords: List[List[float]]
    perimeter: float
    was_fragmented: bool

@dataclass
class OCR:
    ocr_raw: str
    confidence: float
    
@dataclass
class CroppedImage:
    cropped_img: np.ndarray # type: ignore
    
@dataclass
class LineID:
    line_id: str
    
@dataclass(frozen=True)
class Geometry:
    polygon_coords: List[List[float]]
    bounding_box: List[float]
    centroid: List[float]    
    
@dataclass
class Polygons:
    polygon_id: str
    geometry: Geometry
    cropedd_geometry: PaddingGeometry
    line_id: str
    cropped_img: Optional[CroppedImage]
    ocr: OCR

@dataclass
class AllLines:
    line_id: str
    line_bbox: List[float]
    line_centroid: List[float]
    polygon_ids: List[str]

@dataclass
class ImageData:
    polygons: Dict[str, Polygons]
    all_lines: Dict[str, AllLines]
            
@dataclass(frozen=True)
class Metadata:
    image_name: str
    format: str
    img_dims: Dict[str, int]
    dpi: float
    date_creation: str
    color: Optional[str]

@dataclass
class WorkflowDict:
    dict_id: str
    full_img: np.ndarray # type: ignore
    metadata:  Metadata
    image_data: ImageData

