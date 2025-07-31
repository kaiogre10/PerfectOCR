# domain/workflow_job.py
import numpy as np
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class DocumentMetadata:
    """Almacena metadatos inmutables sobre el documento original."""
    doc_dimensions: Dict[int, int]
    dpi_img: int
    original_filename: str

@dataclass
class PolygonGeometry:
    """Almacena los atributos geométricos inmutables de un polígono."""
    polygon_coords: List[List[float]]
    bounding_box: List[float]
    centroid: List[float]
    area: float
    width: int
    height: int

@dataclass
class PreprocessingJob:
    """
    Encapsula la imagen de un polígono y la trazabilidad de su preprocesamiento.
    Este es el "lienzo" para un único polígono.
    """
    img: np.ndarray  # La imagen recortada del polígono que será modificada.
    # Campos para registrar los resultados de cada etapa de corrección.
    moire_stats: Optional[Dict[str, Any]] = None
    sp_stats: Optional[Dict[str, Any]] = None
    gauss_stats: Optional[Dict[str, Any]] = None
    clahe_stats: Optional[Dict[str, Any]] = None
    sharp_stats: Optional[Dict[str, Any]] = None

@dataclass
class PolygonData:
    """
    Representa un único polígono, conteniendo su geometría y su job de preprocesamiento.
    """
    polygon_id: str
    line_id: str
    geometry: PolygonGeometry
    preprocessing: PreprocessingJob
    was_fragmented: bool = False
    text: Optional[str] = None
    confidence: Optional[float] = None

@dataclass
class WorkflowJob:
    """
    El objeto central y único que representa todo el estado del flujo de trabajo.
    Actúa como el "workspace" compartido que es modificado por los managers y workers.
    """
    # === IDENTIFICACIÓN Y ESTADO ===
    job_id: str
    creation_timestamp: float = field(default_factory=time.time)
    current_stage: str = "initialized"

    # === DATOS CENTRALES (EVOLUTIVOS) ===
    # Inicia con la imagen completa, se libera (None) tras la extracción de polígonos.
    full_img: Optional[np.ndarray]
    # La lista de polígonos, que se puebla durante la fase poligonal.
    polygons: List[PolygonData] = field(default_factory=list)

    # === METADATOS Y MÉTRICAS ===
    doc_metadata: DocumentMetadata
    processing_times: Dict[str, float] = field(default_factory=dict)
    history: List[str] = field(default_factory=list)