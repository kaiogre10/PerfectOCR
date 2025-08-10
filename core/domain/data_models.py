# core/domain/workflow_models.py
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from typing import Any

WORKFLOW_SCHEMA = {
    "type": "object",
    "properties": {
        "dict_id": {"type": "string"},
        "full_img": {"anyOf": [{"type": "array"}, {"type": "object", "hasOwnProperty": "shape"}]}, 
        "metadata": {
            "type": "object",
            "properties": {
                "image_name": {"type": "string"},
                "format": {"type": "string"},
                "img_dims": {
                    "type": "object",
                    "properties": {
                        "width": {"type": "integer"},
                        "height": {"type": "integer"},
                    },
                },
                "dpi": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {
                                "x": {"type": "number"},
                                "y": {"type": "null"}
                            }
                        },
                    ],
                },
                "date_creation": {"type": "string"},
                "color": {"type": ["string", "null"]}
            },
        },
        "image_data": {
            "type": "object",
            "properties": {   
                "polygons": {
                    "type": "object",
                    "patternProperties": {
                        "^poly_\\d{4}$": {
                            "type": "object",
                            "properties": {
                                "polygon_id": {"type": "string"},
                                "geometry": {
                                    "type": "object",
                                    "properties": {
                                        "polygon_coords": {
                                            "type": "array",
                                            "items": {
                                                "type": "array",
                                                "items": {"type": "number"},
                                                "minItems": 3,
                                                "maxItems": 4
                                            }
                                        },
                                        "bounding_box": {
                                            "type": "array",
                                            "items": {"type": "number"},
                                            "minItems": 4,
                                            "maxItems": 4
                                        },
                                        "centroid": {
                                            "type": "array",
                                            "items": {"type": "number"},
                                            "minItems": 2,
                                            "maxItems": 2
                                        },
                                    }, 
                                },
                                "cropedd_geometry":{
                                    "type": "object",
                                    "properties": {
                                        "padding_bbox": {
                                            "type": "array",
                                            "items": {"type": "number"},
                                            "minItems": 4,
                                            "maxItems": 4
                                        },
                                        "padd_centroid": {
                                            "type": "array",
                                            "items": {"type": "number"},
                                            "minItems": 2,
                                            "maxItems": 2
                                        },
                                        "padding_coords": {
                                            "type": "array",
                                            "items": {"type": "integer"},
                                            "minItems": 4,
                                            "maxItems": 4
                                        },
                                        "perimeter": {"type": ["number", "null"]},
                                        "was_fragmented": {"type": "boolean"},
                                    }, 
                                },
                                "line_id": {"type": "string"},
                                "cropped_img": {"type": ["object", "null"]},
                                "ocr": {
                                    "type": "object",
                                    "properties": {
                                        "ocr_raw": {"type": ["string", "null"]},
                                        "confidence": {"type": "number", "minimum": 60, "maximum": 100}
                                    },
                                },
                            },
                        },
                    },
                },

            },
        },
    },
    "required": ["dict_id", "metadata", "image_data"]
}

"""""
{"type": "string"}       Debe ser texto
{"type": "integer"}      Debe ser número entero
{"type": "number"}       Debe ser número (puede ser decimal)
{"type": "boolean"}      Debe ser True/False
{"type": "object"}       Debe ser diccionario
{"type": "array"}        Debe ser lista 
"""""

# Mapa de rutas invertido (hoja -> raíz) para ruteo eficiente..
data_paths: Dict[str, List[str]] = {
    "bounding_box": ["geometry", "{poly_id}", "polygons"],
    "centroid": ["geometry", "{poly_id}", "polygons"],
    "confidence": ["ocr", "{poly_id}", "polygons"],
    "cropped_img": ["{poly_id}", "polygons"],
    "line_id": ["{poly_id}", "polygons"],
    "ocr_raw": ["ocr", "{poly_id}", "polygons"],
    "padd_centroid": ["cropedd_geometry", "{poly_id}", "polygons"],
    "padding_bbox": ["cropedd_geometry", "{poly_id}", "polygons"],
    "padding_coords": ["cropedd_geometry", "{poly_id}", "polygons"],
    "perimeter": ["cropedd_geometry", "{poly_id}", "polygons"],
    "polygon_coords": ["geometry", "{poly_id}", "polygons"],
    "polygon_id": ["{poly_id}", "polygons"],
    "was_fragmented": ["cropedd_geometry", "{poly_id}", "polygons"]
}

@dataclass
class PaddingGeometry:
    padding_bbox: List[float]
    padd_centroid: List[float]
    padding_coords: List[List[float]]
    perimeter: Optional[float]
    was_fragmented: bool

@dataclass
class OCR:
    ocr_raw: str
    confidence: float
    
@dataclass
class CroppedImage:
    cropped_img: Optional[np.ndarray[Any, Any]]
        
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
    cropped_img: Optional[CroppedImage]
    ocr: OCR
    line_id: str

@dataclass
class ImageData:
    polygons: Dict[str, Polygons]
                
@dataclass(frozen=True)
class Metadata:
    image_name: str
    format: str
    img_dims: Optional[Dict[str, int]]
    dpi: Optional[Dict[str, Optional[float]]]
    date_creation: str
    color: Optional[str]

@dataclass
class WorkflowDict:
    dict_id: str
    full_img: Optional[np.ndarray[Any, Any]]
    metadata:  Metadata
    image_data: ImageData

