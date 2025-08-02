# domain/workflow_job.py
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class DocumentMetadata:
    """Almacena metadatos inmutables sobre el documento original.
    REFLEJO EXACTO de document_dict["metadata"]"""
    doc_name: str                        # "doc_name": "ejemplo.pdf"
    formato: Optional[str]               # "formato": "PDF"
    img_dims: Dict[str, float]           # "img_dims": {"width": 1240, "height": 1754}
    dpi: Optional[float]                 # "dpi": 300
    fecha_creacion: Optional[str]        # "fecha_creacion": "2023-04-15 10:30:45"
    
@dataclass
class PolygonGeometry:
    """Almacena los atributos geométricos inmutables de un polígono.
    REFLEJO EXACTO de polygon["geometry"]"""
    polygon_coords: List[List[float]]    # "polygon_coords": [[120.0, 230.0], [420.0, 230.0], ...]
    bounding_box: List[float]            # "bounding_box": [120.0, 230.0, 420.0, 260.0]
    centroid: List[float]                # "centroid": [270.0, 245.0]
    width: float                         # "width": 300.0
    height: float                        # "height": 30.0
    
@dataclass
class Polygon:
    """Representa un único polígono.
    REFLEJO EXACTO de document_dict["polygons"]["poly_XXXX"]"""
    polygon_id: str                      # "polygon_id": "poly_0000"
    geometry: PolygonGeometry            # "geometry": {...}
    line_id: str                         # "line_id": "line_0001"
    cropped_img: Optional[np.ndarray]    # "cropped_img": imagen_recortada_0000
    bin_img: Optional[np.ndarray]        # "binarized_poly": polygono binarizado mapeado
    padding_coords: List[float]          # "padding_coords": [115, 225, 425, 265]
    was_fragmented: bool                 # "was_fragmented": False
    perimeter: float                     # "perimeter": 580.0
    text: Optional[str] = None           # "text": "palabra" (añadido por OCR)
    confidence: Optional[float] = None   # "confidence": 95.5 (añadido por OCR)

@dataclass
class LineInfo:
    """La información por línea del documento.
    REFLEJO EXACTO de lines_geometry["line_XXXX"]"""
    line_id: str                         # "line_id": "line_0001"
    bounding_box: List[float]            # "bounding_box": [120.0, 230.0, 1080.0, 260.0]
    centroid: List[float]                # "centroid": [600.0, 245.0]
    polygon_ids: List[str]               # "polygon_ids": ["poly_0000", "poly_0001", "poly_0002"]

@dataclass
class WorkflowJob:
    """REFLEJO COMPLETO del diccionario poligonal y todos sus componentes.
    Contiene:
    - document_dict["metadata"] → doc_metadata
    - document_dict["polygons"] → polygons
    - lines_geometry → lines (convertido a lista para preservar orden)
    - binarized_polygons → binarized_polygons
    """
    job_id: str
    full_img: Optional[np.ndarray]
    # REFLEJO de document_dict["metadata"]
    doc_metadata: DocumentMetadata
    creation_timestamp: float = field(default_factory=time.time)
    current_stage: str = "initialized"
    # REFLEJO de document_dict["polygons"]
    polygons: Dict[str, Polygon] = field(default_factory=dict)
    # REFLEJO de lines_geometry (convertido a lista para preservar orden de lectura)
    lines: List[LineInfo] = field(default_factory=list)
    # REFLEJO de binarized_polygons (diccionario separado)
    binarized_polygons: Dict[str, np.ndarray] = field(default_factory=dict)
    # TIEMPOS DE PROCESAMIENTO
    processing_times: Dict[str, float] = field(default_factory=dict)