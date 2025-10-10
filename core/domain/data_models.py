# core/domain/workflow_models.py
import numpy as np
import pandas as pd # type: ignore
from typing import Dict, List, Optional
from dataclasses import dataclass
from typing import Any

@dataclass
class StructuredTable:
    df: pd.DataFrame
    columns: List[str]
    semantic_types: Optional[List[str]] = None

@dataclass
class CroppedGeometry:
    padd_centroid: np.ndarray[Any, Any]  # shape: (2,)
    padding_coords: np.ndarray[Any, Any]  # shape: (4,) 
    croppy_dims: Dict[str, int]
    
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
    polygon_id: Optional[str]
    geometry: Geometry
    cropedd_geometry: CroppedGeometry
    cropped_img: Optional[CroppedImage]
    its_white: bool
    perimeter: Optional[float]
    ocr_text: Optional[str]
    ocr_confidence: Optional[float]
    key_field: Optional[str]
    semantic_type: str
    was_refined: bool
    
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
    header_line: Optional[str]
    
@dataclass(frozen=True)
class Metadata:
    image_name: str
    img_dims: Dict[str, int]
    date_creation: str

@dataclass
class FullImage:
    full_img: Optional[np.ndarray[Any, np.dtype[np.uint8]]]

@dataclass
class WorkflowDict:
    IDRegistro: str
    full_img: Optional[FullImage]
    metadata: Metadata
    polygons: Dict[str, Polygons]
    all_lines: Dict[str, AllLines]
    
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
    "e": 113,
}

CHAR_FRECUENCY: Dict[str, int] ={
    "a": 998212,
    "e": 751872,
    "r": 631282,
    "s": 574046,
    "i": 454397,
    "o": 407905,
    "n": 399618,
    "c": 284425,
    "t": 264655,
    "l": 235850,
    "m": 235836,
    "d": 209797,
    "u": 171685,
    "b": 144051,
    "p": 140991,
    "á": 102337,
    "g": 90349,
    "í": 67139,
    "h": 62029,
    "z": 61667,
    "f": 57827,
    "v": 50617,
    "é": 47631,
    "j": 38082,
    "q": 24328,
    "ó": 18696,
    "ñ": 14470,
    "x": 9540,
    "y": 8676,
    "ú": 1368,
    "ü": 918,
    "k": 575,
    "w": 84,
    "A": 39,
    "C": 38,
    "P": 31,
    "D": 22,
    "B": 21,
    "S": 19,
    "N": 19,
    "M": 18,
    "R": 17,
    "I": 17,
    "T": 16,
    "V": 16,
    "E": 16,
    "J": 14,
    "L": 11,
    "F": 11,
    "G": 10,
    "O": 5,
    "H": 4,
    "K": 3,
    "U": 3,
    "Q": 2,
    "Y": 2,
    "è": 1,
    "Á": 1,
    "Ó": 1,
}
