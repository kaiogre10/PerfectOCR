# core/domain/workflow_models.py
import numpy as np
import pandas as pd 
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

@dataclass
class StructuredTable:
    df: pd.DataFrame
    columns: List[str]
    semantic_types: Optional[List[str]] = None
    
@dataclass
class CroppedImage:
    cropped_img: Optional[np.ndarray[Any, np.dtype[np.uint8]]]
            
@dataclass
class Geometry:
    polygon_coords: np.ndarray[Any, np.dtype[np.int32]]  # shape: (n_points, 2)
    bounding_box: np.ndarray[Any, np.dtype[np.int16]]    # shape: (4,)
    centroid: np.ndarray[Any, np.dtype[np.float32]]       # shape: (2,)

@dataclass
class Polygons:
    polygon_id: str
    poly_index: int
    geometry: Geometry
    cropped_img: Optional[CroppedImage]
    ocr_text: Optional[str]
    key_field: Optional[List[int] | int]
#    'GlobalData': 0
#    'MontoTotalDocumento': 1
#    'TotalProductos': 2
#    'Subtotal': 3
#    'FolioDocumento': 4
#    "NombreCliente": 5
#    "HeaderWords": 6
#    "RFCProveedor": 7,
#    "MontoIVAGeneral": 8,
#    "FechaDocumento": 9
#    "TelefonoP": 10
#    "CorreoP": 11
#    "DirecciónP": 12

    semantic_clasification: List[int]
        # noise: -1
        # unique: 0
        # descriptive: 1
        # umd: 2
        # code: 3
        # quantitative: 4
        # numeric: 5
        
    cuant_chars: int
    contours: int

@dataclass
class LineGeometry:
    line_centroid: List[float]
    line_bbox: List[float]

@dataclass
class AllLines:
    lineal_id: str
    line_index: int
    text: str
    polygon_ids: List[str]
    polygons_index: List[int]
    line_geometry: LineGeometry
    tabular_line: bool
    header_line: Optional[int]
    footer_line: Optional[int]
    t_cuant: int
    
@dataclass
class Metadata:
    image_name: str
    date_creation: str
    dpi: Optional[int]
    img_dims: Tuple[int, int]

@dataclass
class FullImage:
    full_img: Optional[np.ndarray[Any, Any]]

@dataclass
class WorkflowDict:
    IDRegistro: str
    full_img: Optional[FullImage]
    metadata: Metadata
    polygons: Dict[str, Polygons]
    all_lines: Dict[str, AllLines]
