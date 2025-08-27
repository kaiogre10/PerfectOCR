# core/domain/workflow_models.py
import numpy as np
import pandas as pd # type: ignore
from typing import Dict, List, Optional
from dataclasses import dataclass
from typing import Any

WORKFLOW_SCHEMA: Dict[str , Any] = {
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
                        "size": {"type": "integer"},
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
                                        "minItems": 2,
                                        "maxItems": 2
                                    },
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
                                "poly_dims": {"type": "object",
                                    "properties": {
                                        "poly_width": {"type": "integer"},
                                        "poly_height": {"type": "integer"},
                                    },
                                },
                            }, 
                        },
                        "cropped_img": {"type": ["object", "null"]},
                        "perimeter": {"type": ["number", "null"]},
                        "line_id": {"type": "string"},
                        "ocr_text": {"type": "string"},
                        "ocr_confidence": {"type": ["number", "null"]},
                        "was_fragmented": {"type": "boolean"},
                        "stage": {"type": "string"},
                        "status": {"type": "boolean"},
                    },
                },
            },
        },
        "all_lines": {
            "type": "object",
            "patternProperties": {
                "^line_\\d{4}$": {
                    "type": "object",
                    "properties": {
                        "lineal_id": {"type": "string"},
                        "text": {"type": "string"},
                        "encoded_text": {"type": "array", "items": {"type": "integer"}},
                        "polygon_ids": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "line_bbox": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 4,
                            "maxItems": 4
                        },
                        "line_centroid": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                        "tabular_line": {"type": "boolean"},
                    },
                },
            },
        },
        "tabular_lines": {
           "type": "object",
            "patternProperties": {
            "line_id": {
                "type": "object",
                "properties": {
                "texto": {"type": "string"}
                },
            },
            "header_line": {"type": "boolean"},
            },
        },
    },
}

"""""
{"type": "string"}       Debe ser texto
{"type": "integer"}      Debe ser número entero
{"type": "number"}       Debe ser número (puede ser decimal)
{"type": "boolean"}      Debe ser True/False
{"type": "object"}       Debe ser diccionario
{"type": "array"}        Debe ser lista 
"""""

DENSITY_ENCODER: Dict[str, int] = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    ".": 10,
    ",": 11,
    "$": 12,
    "¢": 13,
    "/": 14,
    "#": 15,
    "°": 16,
    "(": 17,
    ")": 18,
    "%": 19,
    "—": 20,
    "<": 21,
    ">": 22,
    "+": 23,
    "-": 24,
    "=": 25,
    "*": 26,
    "^": 27,
    "\"": 28,
    ";": 29,
    "\\": 30,
    "|": 31,
    "[": 32,
    "]": 33,
    "{": 34,
    "}": 35,
    "@": 36,
    "&": 37,
    "_": 38,
    "¿": 39,
    "?": 40,
    "¡": 41,
    "!": 42,
    "~": 43,
    "`": 44,
    "'": 45,
    ":": 46,
    "©": 47,
    "®": 48,
    "™": 49,
    "Ó": 50,
    "Á": 51,
    "Ú": 52,
    "Ü": 53,
    "Ñ": 54,
    "W": 55,
    "X": 56,
    "Z": 57,
    "Y": 58,
    "Q": 59,
    "U": 60,
    "K": 61,
    "H": 62,
    "O": 63,
    "G": 64,
    "F": 65,
    "L": 66,
    "J": 67,
    "E": 68,
    "V": 69,
    "T": 70,
    "I": 71,
    "R": 72,
    "M": 73,
    "N": 74,
    "S": 75,
    "B": 76,
    "D": 77,
    "P": 78,
    "C": 79,
    "A": 80,
    "w": 81,
    "k": 82,
    "ü": 83,
    "ú": 84,
    "y": 85,
    "x": 86,
    "ñ": 87,
    "ó": 88,
    "q": 89,
    "j": 90,
    "é": 91,
    "v": 92,
    "f": 93,
    "z": 94,
    "h": 95,
    "í": 96,
    "g": 97,
    "á": 98,
    "p": 99,
    "b": 100,
    "u": 101,
    "d": 102,
    "m": 103,
    "l": 104,
    "t": 105,
    "c": 106,
    "n": 107,
    "o": 108,
    "i": 109,
    "s": 110,
    "r": 111,
    "a": 112,
    "e": 113
}

@dataclass
class StructuredTable:
    df: pd.DataFrame
    columns: List[str]
    semantic_types: Optional[List[str]] = None
    
@dataclass
class CroppedGeometry:
    padd_centroid: np.ndarray[Any, Any]  # shape: (2,)
    padding_coords: np.ndarray[Any, Any]  # shape: (4,) 
    poly_dims: Dict[str, int]
    
@dataclass
class CroppedImage:
    cropped_img: np.ndarray[Any, np.dtype[np.uint8]]
            
@dataclass(frozen=True)
class Geometry:
    polygon_coords: np.ndarray[Any, Any]  # shape: (n_points, 2)
    bounding_box: np.ndarray[Any, Any]    # shape: (4,)
    centroid: np.ndarray[Any, Any]        # shape: (2,)    
    
@dataclass
class Polygons:
    polygon_id: str
    geometry: Geometry
    cropedd_geometry: CroppedGeometry
    cropped_img: Optional[CroppedImage]
    perimeter: Optional[float]
    line_id: str
    ocr_text: Optional[str]
    ocr_confidence: Optional[float]
    was_fragmented: bool
    status: bool
    stage: str

@dataclass
class TabularLines:
    lineal_id: str
    complete_text: str
    header_line: bool

@dataclass
class LineGeometry:
    line_centroid: List[float]
    line_bbox: List[float]
    
@dataclass
class AllLines:
    lineal_id: str
    text: str
    encoded_text: List[int]
    polygon_ids: List[str]
    line_geometry: LineGeometry
    tabular_line: bool

@dataclass(frozen=True)
class Metadata:
    image_name: str
    format: str
    img_dims: Dict[str, int]
    dpi: Optional[Dict[str, Optional[float]]]
    date_creation: str
    color: Optional[str]

@dataclass
class WorkflowDict:
    dict_id: str
    full_img: Optional[np.ndarray[Any, np.dtype[np.uint8]]]
    metadata:  Dict[str, Metadata]
    polygons: Dict[str, Polygons]
    all_lines: Dict[str, AllLines]
    tabular_lines: Dict[str, TabularLines]
