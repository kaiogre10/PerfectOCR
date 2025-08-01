# domain/workflow_job.py
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class DocumentMetadata:
    """Almacena metadatos inmutables sobre el documento original."""
    doc_name: str
    doc_dimensions: Dict[int, int]
    format: str
    dpi_img: int
    date: str
    
@dataclass
class PolygonGeometry:
    """Almacena los atributos geométricos inmutables de un polígono."""
    polygon_coords: List[Tuple[float, float]]
    bounding_box: List[float]
    centroid: Tuple[float, float]
    width: int
    height: int
    cropped_img: np.ndarray
    padding_coords: List[Tuple[float, float]]
    was_fragmented: bool
    perimeter: float
    line_id: int
    bin_img: np.ndarray
    
@dataclass
class PreprocessingJob:
    """
    Encapsula la imagen de un polígono y la trazabilidad de su preprocesamiento.
    Este es el "lienzo" para un único polígono.
    """
    img: np.ndarray
    moire_stats: Optional[Dict[str, Any]] = None
    sp_stats: Optional[Dict[str, Any]] = None
    gauss_stats: Optional[Dict[str, Any]] = None
    clahe_stats: Optional[Dict[str, Any]] = None
    sharp_stats: Optional[Dict[str, Any]] = None
    
@dataclass
class Polygon:
    """
    Representa un único polígono, conteniendo su geometría y su job de preprocesamiento.
    """
    line_id: str
    geometry: PolygonGeometry
    preprocessing: PreprocessingJob
    ocr: Optional[str] = None
    confidence: Optional[float] = None

@dataclass
class LineInfo:
    """La información por línea del documento"""
    line_id: str
    bounding_box: List[float]
    centroid: Tuple[float, float]
    polygon_ids: List[str]

@dataclass
class WorkflowJob:
    job_id: str
    full_img: Optional[np.ndarray]
    doc_metadata: DocumentMetadata
    creation_timestamp: float = field(default_factory=time.time)
    current_stage: str = "initialized"
    polygons: Dict[str, Polygon] = field(default_factory=dict)
    lines: List[LineInfo] = field(default_factory=list)
    processing_times: Dict[str, float] = field(default_factory=dict)