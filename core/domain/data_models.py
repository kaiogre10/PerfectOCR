# core/domain/data_models.py
import numpy as np
import pandas as pd # type: ignore
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
    
@dataclass(slots=True)
class CroppedImage:
    cropped_img: Optional[np.ndarray[Any, np.dtype[np.uint8]]]
            
@dataclass(slots=True)
class Geometry:
    polygon_coords: np.ndarray[Any, np.dtype[np.int32]]  # shape: (n_points, 1, 4)
    bounding_box: np.ndarray[Any, np.dtype[np.float32]]    # shape: (4,)
    centroid: np.ndarray[Any, np.dtype[np.float32]]       # shape: (2,)

@dataclass(slots=True)
class Polygons:
    polygon_id: str
    poly_index: int
    geometry: Geometry
    cropped_img: Optional[CroppedImage]
    ocr_text: Optional[str]
    key_field: Optional[List[int]]

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

@dataclass(slots=True)
class LineGeometry:
    line_centroid: List[float]
    line_bbox: List[float]

@dataclass(slots=True)
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
    
@dataclass(slots=True)
class Metadata:
    image_name: str
    date_creation: str
    dpi: Optional[int]
    img_dims: Tuple[int, int]
    binary: bool

@dataclass(slots=True)
class FullImage:
    full_img: Optional[np.ndarray[Any, np.dtype[np.uint8]]]

@dataclass(slots=True)
class StructuredData:
    df_table: Optional[pd.DataFrame]
    global_data: Dict[str, Any]

@dataclass(slots=True)
class WorkflowData:
    IDRegistro: Optional[str]
    full_img: Optional[FullImage]
    metadata: Optional[Metadata]
    polygons: Optional[Dict[str, Polygons]]
    all_lines: Optional[Dict[str, AllLines]]
    table_data: Optional[StructuredData]
