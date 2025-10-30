# core/domain/workflow_models.py
import numpy as np
import pandas as pd # type: ignore
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class StructuredTable:
    df: pd.DataFrame
    columns: List[str]
    semantic_types: Optional[List[str]] = None

@dataclass
class CroppedGeometry:
    padd_centroid: np.ndarray[Any, Any]  # shape: (2,)
    padding_coords: np.ndarray[Any, Any]  # shape: (4,) 
    croppy_dims: Dict[str, int]
    
@dataclass
class CroppedImage:
    cropped_img: Optional[np.ndarray[Any, np.dtype[np.uint8]]]
            
@dataclass
class Geometry:
    polygon_coords: np.ndarray[Any, Any]  # shape: (n_points, 2)
    bounding_box: np.ndarray[Any, Any]    # shape: (4,)
    centroid: np.ndarray[Any, Any]        # shape: (2,)

@dataclass
class Polygons:
    polygon_id: Optional[str]
    geometry: Geometry
    cropedd_geometry: CroppedGeometry
    cropped_img: Optional[CroppedImage]
    perimeter: Optional[float]
    ocr_text: Optional[str]
    ocr_confidence: Optional[float]
    key_field: Optional[str]
    semantic_clasification: int
        # quantitative: 2
        # numeric: 1
        # descriptive: 0
        # code: -1
        # umd: -2
    was_refined: bool
    binarized: bool
    
@dataclass
class LineGeometry:
    line_centroid: List[float]
    line_bbox: List[float]
    
@dataclass
class AllLines:
    lineal_id: str
    text: str
    polygon_ids: List[str]
    line_geometry: LineGeometry
    tabular_line: bool
    header_line: Optional[str]
    footer_line: Optional[str]
    
@dataclass
class Metadata:
    image_name: str
    img_dims: Dict[str, int]
    date_creation: str

@dataclass
class FullImage:
    full_img: Optional[np.ndarray[Any, np.dtype[np.uint8]]]

@dataclass
class WorkflowDict:
    IDRegistro: str
    full_img: Optional[FullImage]
    metadata: Metadata
    polygons: Dict[str, Polygons]
    all_lines: Dict[str, AllLines]
