# core/domain/workflow_job.py
import numpy as np
import time
import uuid
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

class ProcessingStage(Enum):
    """Estados del procesamiento para tracking preciso"""
    INITIALIZED = "initialized"
    IMAGE_LOADED = "image_loaded"
    GEOMETRY_DETECTED = "geometry_detected"
    POLYGONS_EXTRACTED = "polygons_extracted"
    LINES_RECONSTRUCTED = "lines_reconstructed"
    PREPROCESSING_COMPLETE = "preprocessing_complete"
    OCR_COMPLETE = "ocr_complete"
    VECTORIZATION_COMPLETE = "vectorization_complete"
    ERROR = "error"

class ProcessingStatus(Enum):
    """Estado general del job"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass(frozen=True)
class ImageDimensions:
    """Dimensiones de imagen inmutables para evitar errores"""
    width: int
    height: int
    
    def __post_init__(self):
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Dimensiones deben ser positivas")
    
    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

@dataclass(frozen=True)
class DocumentMetadata:
    """Metadatos del documento con validación robusta"""
    doc_name: str
    img_dims: ImageDimensions
    formato: Optional[str] = None
    dpi: Optional[float] = None
    color: Optional[str] = None
    date_creation: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.doc_name:
            raise ValueError("Nombre de documento requerido")
        if self.dpi is not None and self.dpi <= 0:
            raise ValueError("DPI debe ser positivo")

@dataclass
class BoundingBox:
    """Bounding box con validación"""
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
    def centroid(self) -> Tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)

@dataclass
class PolygonGeometry:
    """Geometría de polígono con métodos útiles"""
    polygon_coords: List[Tuple[float, float]]
    bounding_box: BoundingBox
    centroid: Tuple[float, float]
    perimeter: Optional[float] = None

    def __post_init__(self):
        if len(self.polygon_coords) < 3:
            raise ValueError("Polígono debe tener al menos 3 puntos")

@dataclass
class Polygon:
    """Polígono con validación y métodos útiles"""
    polygon_id: str
    geometry: PolygonGeometry
    line_id: Optional[str] = None
    cropped_img: Optional[np.ndarray] = None  # type: ignore
    padding_coords: Optional[List[float]] = None
    was_fragmented: bool = False
    text: Optional[str] = None
    confidence: Optional[float] = None
    
    def __post_init__(self):
        if not self.polygon_id.startswith("poly_"):
            raise ValueError("ID de polígono debe empezar con 'poly_'")
        if self.line_id is not None and not self.line_id.startswith("line_"):
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
    bounding_box: BoundingBox
    centroid: Tuple[float, float]
    polygon_ids: List[str]
    
    def __post_init__(self):
        if not self.line_id.startswith("line_"):
            raise ValueError("ID de línea debe empezar con 'line_'")
        if not self.polygon_ids:
            raise ValueError("Línea debe tener al menos un polígono")
    
    @property
    def width(self) -> float:
        return self.bounding_box.width
    
    @property
    def height(self) -> float:
        return self.bounding_box.height

@dataclass
class WorkflowJob:
    """Estructura principal optimizada para máxima eficiencia"""
    # Identificación única
    job_id: str = field(default_factory=lambda: f"job_{uuid.uuid4().hex[:8]}")
    
    # Imagen principal (referencia, no copia)
    full_img: Optional[np.ndarray] = None  # type: ignore
    
    # Metadatos inmutables
    doc_metadata: Optional[DocumentMetadata] = None
    
    # Estado del procesamiento
    current_stage: ProcessingStage = ProcessingStage.INITIALIZED
    status: ProcessingStatus = ProcessingStatus.PENDING
    
    # Timestamps precisos
    creation_timestamp: float = field(default_factory=time.time)
    stage_timestamps: Dict[ProcessingStage, float] = field(default_factory=dict)
    
    # Datos estructurados
    polygons: Dict[str, Polygon] = field(default_factory=dict)
    lines: List[LineInfo] = field(default_factory=list)
    
    # Métricas de rendimiento
    processing_times: Dict[str, float] = field(default_factory=dict)
    error_log: List[str] = field(default_factory=list)
    
    def add_polygon(self, polygon: Polygon) -> None:
        """Añade polígono con validación"""
        if polygon.polygon_id in self.polygons:
            raise ValueError(f"Polígono {polygon.polygon_id} ya existe")
        self.polygons[polygon.polygon_id] = polygon
    
    def add_line(self, line: LineInfo) -> None:
        """Añade línea con validación"""
        if any(l.line_id == line.line_id for l in self.lines):
            raise ValueError(f"Línea {line.line_id} ya existe")
        self.lines.append(line)
    
    def update_stage(self, stage: ProcessingStage) -> None:
        """Actualiza etapa con timestamp"""
        self.current_stage = stage
        self.stage_timestamps[stage] = time.time()
    
    def add_error(self, error: str) -> None:
        """Añade error al log"""
        self.error_log.append(f"[{datetime.now()}] {error}")
        self.status = ProcessingStatus.FAILED
    
    def get_polygons_by_line(self, line_id: str) -> List[Polygon]:
        """Obtiene polígonos de una línea específica"""
        return [p for p in self.polygons.values() if p.line_id == line_id]
    
    def get_polygons_with_text(self) -> List[Polygon]:
        """Obtiene polígonos que tienen texto"""
        return [p for p in self.polygons.values() if p.has_text]
    
    def get_high_confidence_polygons(self) -> List[Polygon]:
        """Obtiene polígonos con alta confianza"""
        return [p for p in self.polygons.values() if p.is_high_confidence]
    
    @property
    def total_polygons(self) -> int:
        return len(self.polygons)
    
    @property
    def total_lines(self) -> int:
        return len(self.lines)
    
    @property
    def has_errors(self) -> bool:
        return len(self.error_log) > 0
    
    @property
    def is_completed(self) -> bool:
        return self.status == ProcessingStatus.COMPLETED
    
    @property
    def processing_duration(self) -> float:
        if self.status == ProcessingStatus.COMPLETED:
            return time.time() - self.creation_timestamp
        return 0.0