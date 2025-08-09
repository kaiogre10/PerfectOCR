# core/domain/workflow_dict.py
import jsonschema
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

class DataFormatter:
    """
    Válvula de entrada/salida para todas las operaciones del dict.
    Los workers NO tocan directamente el dict_id, solo pasan por aquí.
    """

    def __init__(self):
        self.dict_id: Dict[str, Any] = {}
        self.schema = self.WORKFLOW_DICT
    
    def create_dict(self, dict_id: str, full_img: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """Crea un nuevo dict con validación automática"""
        try:
            self.dict_id = {
                "dict_id": dict_id,
                "full_img": full_img,
                "metadata": self._validate_metadata(metadata),
                "image_data": {
                    "polygons": {},
                    "all_lines": {}
                }
            }
            self._validate_structure()
            return True
        except Exception as e:
            print(f"Error creando dict: {e}")
            return False
    
    def add_polygon(self, polygon_data: Dict[str, Any]) -> bool:
        """Agrega polígono con validación automática usando dataclass"""
        try:
            # 1. Construir objetos dataclass para validación y estructura
            geometry_obj = Geometry(
                polygon_coords=polygon_data["geometry"]["polygon_coords"],
                bounding_box=polygon_data["geometry"]["bounding_box"],
                centroid=polygon_data["geometry"]["centroid"]
            )
            
            padding_obj = PaddingGeometry(
                padding_bbox=polygon_data["cropedd_geometry"]["padding_bbox"],
                padd_centroid=polygon_data["cropedd_geometry"]["padd_centroid"],
                padding_coords=polygon_data["cropedd_geometry"]["padding_coords"],
                perimeter=polygon_data["cropedd_geometry"]["perimeter"],
                was_fragmented=polygon_data["cropedd_geometry"]["was_fragmented"]
            )

            ocr_obj = OCR(
                ocr_raw=polygon_data.get("ocr", {}).get("ocr_raw", ""),
                confidence=polygon_data.get("ocr", {}).get("confidence", 0.0)
            )

            poly_obj = Polygons(
                polygon_id=polygon_data["polygon_id"],
                geometry=geometry_obj,
                cropedd_geometry=padding_obj,
                line_id=polygon_data["line_id"],
                cropped_img=polygon_data.get("cropped_img"),
                ocr=ocr_obj
            )

            # 2. Convertir a dict para insertar en el contenedor universal
            poly_dict = asdict(poly_obj)
            poly_id = poly_dict["polygon_id"]
            
            # 3. Insertar y validar contra el schema
            self.dict_id["image_data"]["polygons"][poly_id] = poly_dict
            self._validate_structure()
            return True
        except (KeyError, TypeError) as e:
            print(f"Error de datos agregando polígono: {e}")
            return False
    
    def add_line(self, line_data: Dict[str, Any]) -> bool:
        """Agrega línea con validación automática usando dataclass"""
        try:
            # 1. Construir objeto dataclass
            line_obj = AllLines(
                line_id=line_data["line_id"],
                line_bbox=line_data["line_bbox"],
                line_centroid=line_data["line_centroid"],
                polygon_ids=line_data["polygon_ids"]
            )
            
            # 2. Convertir a dict
            line_dict = asdict(line_obj)
            line_id = line_dict["line_id"]

            # 3. Insertar y validar
            self.dict_id["image_data"]["all_lines"][line_id] = line_dict
            self._validate_structure()
            return True
        except (KeyError, TypeError) as e:
            print(f"Error de datos agregando línea: {e}")
            return False
    
    def update_polygon_ocr(self, poly_id: str, ocr_text: str, confidence: float) -> bool:
        """Actualiza OCR de un polígono específico usando dataclass"""
        try:
            if poly_id not in self.dict_id["image_data"]["polygons"]:
                raise ValueError(f"Polígono {poly_id} no existe")
            
            # 1. Construir objeto OCR
            ocr_obj = OCR(
                ocr_raw=str(ocr_text),
                confidence=float(confidence)
            )
            
            # 2. Actualizar, convertir a dict y validar
            self.dict_id["image_data"]["polygons"][poly_id]["ocr"] = asdict(ocr_obj)
            self._validate_structure()
            return True
        except (KeyError, TypeError, ValueError) as e:
            print(f"Error actualizando OCR: {e}")
            return False
    
    def get_dict_data(self) -> Dict[str, Any]:
        """Devuelve copia completa del dict"""
        return self.dict_id.copy()
    
    def get_polygons(self) -> Dict[str, Any]:
        """Devuelve solo los polígonos"""
        return self.dict_id["document_dict"]["polygons"].copy()
    
    def get_lines(self) -> Dict[str, Any]:
        """Devuelve solo las líneas"""
        return self.dict_id["document_dict"]["all_lines"].copy()
    
    def get_metadata(self) -> Dict[str, Any]:
        """Devuelve solo los metadatos"""
        return self.dict_id["document_dict"]["metadata"].copy()
    
    def _validate_structure(self) -> None:
        """Valida estructura completa contra schema JSON"""
        jsonschema.validate(self.dict_id, self.schema)
    
    def _validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Transforma y valida metadatos"""
        return {
            "image_name": str(metadata.get("image_name", "")),
            "format": str(metadata.get("format", "")),
            "img_dims": {
                "width": int(metadata.get("img_dims", {}).get("width", 0)),
                "height": int(metadata.get("img_dims", {}).get("height", 0))
            },
            "dpi": float(metadata["dpi"]) if metadata.get("dpi") is not None else None,
            "date_creation": metadata.get("date_creation", datetime.now().isoformat()),
            "color": str(metadata["color"]) if metadata.get("color") is not None else None
        }
    
    def _transform_polygon(self, polygon_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transforma datos simples de polígono a estructura completa"""
        geometry = polygon_data.get("geometry", {})
        
        return {
            "polygon_id": str(polygon_data["polygon_id"]),
            "geometry": {
                "polygon_coords": [[float(x), float(y)] for x, y in geometry["polygon_coords"]],
                "bounding_box": [float(x) for x in geometry["bounding_box"]],
                "centroid": [float(x) for x in geometry["centroid"]],
                "width": float(geometry.get("width", 0)),
                "height": float(geometry.get("height", 0)),
                "padding_coords": [int(x) for x in geometry.get("padding_coords", [0, 0, 0, 0])],
                "perimeter": float(geometry.get("perimeter", 0))
            },
            "line_id": str(polygon_data["line_id"]) if polygon_data.get("line_id") else None,
            "cropped_img": polygon_data.get("cropped_img"),
            "was_fragmented": bool(polygon_data.get("was_fragmented", False)),
            "ocr": polygon_data.get("ocr", {"ocr_raw": "", "confidence": 0.0})
        }
    
    def _transform_line(self, line_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transforma datos simples de línea a estructura completa"""
        return {
            "line_id": str(line_data["line_id"]),
            "line_bounding_box": [float(x) for x in line_data["line_bounding_box"]],
            "line_centroid": [float(x) for x in line_data["line_centroid"]],
            "polygon_ids": [str(x) for x in line_data["polygon_ids"]]
        }

    """""
    {"type": "string"}       Debe ser texto
    {"type": "integer"}      Debe ser número entero
    {"type": "number"}       Debe ser número (puede ser decimal)
    {"type": "boolean"}      Debe ser True/False
    {"type": "object"}       Debe ser diccionario
    {"type": "array"}        Debe ser lista 
    """""

    WORKFLOW_DICT = {
        "type": "object",
        "properties": {
            "dict_id": {"type": "string"},
            "full_img": {"type": "object"},  # np.ndarray se serializa como object
            "metadata": {
                "type": "object",
                "properties": {
                    "image_name": {"type": "string"},
                    "format": {"type": "string"},
                    "img_dims": {
                        "type": "object",
                        "properties": {
                            "width": {"type": "integer"},
                            "height": {"type": "integer"}
                        },
                        "required": ["width", "height"]
                    },
                    "dpi": {"type": ["number", "null"]},
                    "date_creation": {"type": "string"},
                    "color": {"type": ["string", "null"]}
                },
                "required": ["image_name", "format", "img_dims"]
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
                                                    "minItems": 2,
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
                                        "required": ["polygon_coords", "bounding_box", "centroid"] 
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
                                        "required": ["padding_coords", "padd_centroid", "perimeter", "was_fragmented"] 
                                    },
                                    "line_id": {"type": "string"},
                                    "cropped_img": {"type": ["object", "null"]},
                                    "ocr": {
                                        "type": "object",
                                        "properties": {
                                            "ocr_raw": {"type": ["string", "null"]},
                                            "confidence": {"type": "number", "minimum": 60, "maximum": 100}
                                        },
                                        "required": ["ocr_raw", "confidence"]
                                    }
                                },
                                "required": ["polygon_id", "geometry", "cropedd_geometry", "line_id"]
                            },
                        },
                    },
                    "all_lines": {
                        "type": "object",
                        "patternProperties": {
                            "^line_\\d{4}$": {
                                "type": "object",
                                "properties": {
                                    "line_id": {"type": "string"},
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
                                        "maxItems": 2
                                    },
                                    "polygon_ids": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                },
                                "required": ["line_id", "line_bbox", "line_centroid", "polygon_ids"]
                            }
                        }
                    }
                },
                "required": ["polygons", "all_lines"]
            },
        },
        "required": ["dict_id", "full_img", "metadata", "image_data"]
    },
    
@dataclass
class PaddingGeometry:
    padding_bbox: List[float]
    padd_centroid: List[float]
    padding_coords: List[List[float]]
    perimeter: float
    was_fragmented: bool

@dataclass
class OCR:
    ocr_raw: str
    confidence: float
    
@dataclass
class CroppedImage:
    cropped_img: np.ndarray # type: ignore
    
@dataclass
class LineID:
    line_id: str
    
@dataclass(frozen=True)
class Geometry:
    polygon_coords: List[int]
    bounding_box: List[float]
    centroid: List[float]    
    
@dataclass
class Polygons:
    polygon_id: str
    geometry: Geometry
    cropedd_geometry: PaddingGeometry
    line_id: str
    cropped_img: Optional[CroppedImage]
    ocr: OCR

@dataclass
class AllLines:
    line_id: str
    line_bbox: List[float]
    line_centroid: List[float]
    polygon_ids: List[str]

@dataclass
class ImageData:
    polygons: Dict[str, Polygons]
    all_lines: Dict[str, AllLines]
            
@dataclass(frozen=True)
class Metadata:
    image_name: str
    format: str
    img_dims: Dict[str, int]
    dpi: float
    date_creation: str
    color: Optional[str]

@dataclass
class WorkflowDict:
    dict_id: str
    full_img: np.ndarray # type: ignore
    metadata:  Metadata
    image_data: ImageData