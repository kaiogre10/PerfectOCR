# domain/data_models.py
import numpy as np
import pandas as pd # type: ignore
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
    
@dataclass(slots=True)
class Polygons:
    polygon_id: str
    poly_index: int
    bounding_box: List[int]
    centroid: List[float]
    cropped_img: Optional[np.ndarray[Any, np.dtype[np.uint8]]]
    ocr_text: Optional[str]
    key_field: Optional[List[int]]
    semantic_clasification: List[int]
    cuant_chars: int

@dataclass(slots=True)
class AllLines:
    lineal_id: str
    line_index: int
    text: str
    polygon_ids: List[str]
    polygons_index: List[int]
    line_centroid: List[float]
    line_bbox: List[float]
    tabular_line: bool
    header_line: Optional[int]
    footer_line: Optional[int]
    
@dataclass(slots=True)
class Metadata:
    image_name: str
    img_dims: Tuple[int, int]

@dataclass(slots=True)
class StructuredData:
    df_table: Optional[pd.DataFrame]
    global_data: Dict[str, Any]

@dataclass(slots=True)
class WorkflowData:
    full_img: Optional[np.ndarray[Any, np.dtype[np.uint8]]]
    metadata: Optional[Metadata]
    polygons: Optional[Dict[str, Polygons]]
    all_lines: Optional[Dict[str, AllLines]]
    table_data: Optional[StructuredData]

@dataclass(slots=True)
class Payload:
    payload: Optional[str]
    buff_size: int