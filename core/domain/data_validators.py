# core/domain/data_validators.py
from pydantic import BaseModel, ConfigDict
from typing import Dict, Any, Optional, List
import numpy as np

# Configuración base para tipos especiales
class BaseValidator(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # Permite np.ndarray
        validate_assignment=True,      # Valida al asignar
        extra='forbid'                 # Prohíbe campos extra
    )

# Modelos específicos para Metadata
class ImageDimensionsValidator(BaseValidator):
    width: int
    height: int
    size: int

class DpiValidator(BaseValidator):
    x: float
    y: Optional[float] = None

class MetadataValidator(BaseValidator):
    image_name: str
    format: str
    img_dims: ImageDimensionsValidator
    dpi: Optional[DpiValidator] = None
    date_creation: Optional[str] = None
    color: Optional[str] = None

# Geometría
class GeometryValidator(BaseValidator):
    polygon_coords: Any  # np.ndarray
    bounding_box: Any    # np.ndarray
    centroid: Any        # np.ndarray

class CroppedGeometryValidator(BaseValidator):
    padd_centroid: Any   # np.ndarray
    padding_coords: Any  # np.ndarray
    poly_dims: Dict[str, int]

class CroppedImageValidator(BaseValidator):
    cropped_img: Any     # np.ndarray

# Polígonos
class PolygonsValidator(BaseValidator):
    polygon_id: str
    geometry: GeometryValidator
    cropedd_geometry: CroppedGeometryValidator
    cropped_img: Optional[CroppedImageValidator] = None
    perimeter: Optional[float] = None
    line_id: str
    ocr_text: Optional[str] = None
    ocr_confidence: Optional[float] = None
    was_fragmented: bool
    status: bool
    stage: str

# Líneas
class LineGeometryValidator(BaseValidator):
    line_centroid: List[float]
    line_bbox: List[float]

class AllLinesValidator(BaseValidator):
    lineal_id: str
    text: str
    encoded_text: List[int]
    polygon_ids: List[str]
    line_geometry: LineGeometryValidator
    tabular_line: bool
    header_line: bool

# Validador principal - CORREGIDO
class WorkflowValidator(BaseValidator):
    dict_id: str
    full_img: Optional[np.ndarray[Any, np.dtype[np.uint8]]]
    metadata: MetadataValidator  # ← CORRECTO: Espera MetadataValidator
    polygons: Dict[str, Any] = {}
    all_lines: Dict[str, Any] = {}