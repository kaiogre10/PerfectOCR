# core/domain/workflow_manager.py
import jsonschema
import numpy as np
from typing import Dict, Any
from datetime import datetime
from dataclasses import asdict

class DataFormatter:
    """
    Válvula de entrada/salida para todas las operaciones del dict.
    Los workers NO tocan directamente el dict_id, solo pasan por aquí.
    """

    def __init__(self):
        self.dict_id: Dict[str, Any] = {}
        self.WORKFLOW_DICT: Dict[str, Any]
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
            
            line_dict = asdict(line_obj)
            line_id = line_dict["line_id"]

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
        return self.dict_id["image_data"]["polygons"].copy()
    
    def get_lines(self) -> Dict[str, Any]:
        """Devuelve solo las líneas"""
        return self.dict_id["image_data"]["all_lines"].copy()
    
    def get_metadata(self) -> Dict[str, Any]:
        """Devuelve solo los metadatos"""
        return self.dict_id["metadata"].copy()
    
    def _validate_structure(self) -> None:
        """Valida estructura completa contra schema JSON"""
        jsonschema.validate(self.dict_id, self.schema)
    
    def _validate_image_data(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        polygon_data = image_data.get("polygon", {})
        geometry = polygon_data.get("geometry", {})
        cropedd_geometry = polygon_data.get("cropedd_geometry", {})
        ocr = polygon_data.get("ocr", {})
        line_data = image_data.get("line", {})
        
        return {
            "polygons": {
                "polygon_id": str(polygon_data.get("polygon_id", "")),
                "geometry": {
                    "polygon_coords": [
                        [float(coord[0]), float(coord[1])]
                        for coord in geometry.get("polygon_coords", [])
                    ],
                    "bounding_box": [float(x) for x in geometry.get("bounding_box", [0.0, 0.0, 0.0, 0.0])],
                    "centroid": [float(x) for x in geometry.get("centroid", [0.0, 0.0])],
                },
                "cropedd_geometry": {
                    "padding_bbox": [float(x) for x in cropedd_geometry.get("padding_bbox", [0.0, 0.0, 0.0, 0.0])],
                    "padd_centroid": [float(x) for x in cropedd_geometry.get("padd_centroid", [0.0, 0.0])],
                    "padding_coords": [
                        [int(coord[0]), int(coord[1])]
                        for coord in cropedd_geometry.get("padding_coords", [])
                    ],
                    "perimeter": float(cropedd_geometry.get("perimeter", 0.0)),
                    "was_fragmented": bool(cropedd_geometry.get("was_fragmented", False)),
                },
                "line_id": str(polygon_data.get("line_id", "")),
                "cropped_img": polygon_data.get("cropped_img"),
                "ocr": {
                    "ocr_raw": ocr.get("ocr_raw", ""),
                    "confidence": float(ocr.get("confidence", 60.0)),
                }
            },
            "all_lines": {
                "line_id": str(line_data.get("line_id", "")),
                "line_bbox": [float(x) for x in line_data.get("line_bbox", [0.0, 0.0, 0.0, 0.0])],
                "line_centroid": [float(x) for x in line_data.get("line_centroid", [0.0, 0.0])],
                "polygon_ids": [str(x) for x in line_data.get("polygon_ids", [])]
            }
        }

    def _validate_line_data(self, line_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transforma datos simples de línea a estructura completa conforme al schema"""
        return {
            "line_id": str(line_data.get("line_id", "")),
            "line_bbox": [float(x) for x in line_data.get("line_bbox", [0.0, 0.0, 0.0, 0.0])],
            "line_centroid": [float(x) for x in line_data.get("line_centroid", [0.0, 0.0])],
            "polygon_ids": [str(x) for x in line_data.get("polygon_ids", [])]
        }
    
    def _transform_line(self, line_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transforma datos simples de línea a estructura completa"""
        return {
            "line_id": str(line_data["line_id"]),
            "line_bounding_box": [float(x) for x in line_data["line_bounding_box"]],
            "line_centroid": [float(x) for x in line_data["line_centroid"]],
            "polygon_ids": [str(x) for x in line_data["polygon_ids"]]
        }

WORKFLOW_SCHEMA = {
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
}
