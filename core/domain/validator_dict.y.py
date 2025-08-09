# core/domain/workflow_validator.py
import jsonschema
from typing import Any, Dict, Optional
import numpy as np
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime

class DataValidator:
    """
    Válvula de entrada/salida para todas las operaciones del dict.
    Los workers NO tocan directamente el dict_id, solo pasan por aquí.
    """
    def __init__(self):
        self.dict_id: Dict[str, Any] = {}
        self.WORKFLOW_DICT: Dict[str, Any]
        self.schema = self.WORKFLOW_DICT

    def update_data(
        self,
        path: str,
        value: Any,
        *,
        poly_id: Optional[str] = None,
        line_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Actualización universal sin creaciones implícitas. Trabaja 'sobre original'.
        - Siempre retorna estado, no lanza fuera.
        - Valida/clamp para cumplir SCHEMA; si el cambio rompería el schema, revierte localmente.
        - Nunca crea contenedores ni claves que no existan (flujo secuencial).
        """
        status: Dict[str, Any] = {
            "success": True,
            "applied": False,
            "sanitized": False,
            "reason": None,
            "provided_value": value,
            "stored_value": None,
            "path": path,
            "poly_id": poly_id,
            "line_id": line_id,
        }

        try:
            if not isinstance(path, str) or "." not in path:
                status["reason"] = "invalid_path"
                return status

            root, *rest = path.split(".")

            # METADATA
            if root == "metadata":
                md = self.dict_id.get("metadata")
                if not isinstance(md, dict):
                    status["reason"] = "not_found"
                    return status

                # metadata.img_dims.width / height
                if rest[:2] == ["img_dims", "width"]:
                    if "img_dims" not in md or not isinstance(md["img_dims"], dict):
                        status["reason"] = "not_found"
                        return status
                    prev = md["img_dims"].get("width")
                    md["img_dims"]["width"] = int(value)
                    try:
                        self._validate_structure()
                        status["applied"] = True
                        status["stored_value"] = md["img_dims"]["width"]
                    except Exception:
                        md["img_dims"]["width"] = prev
                        status["reason"] = "schema_validation_failed"
                    return status

                if rest[:2] == ["img_dims", "height"]:
                    if "img_dims" not in md or not isinstance(md["img_dims"], dict):
                        status["reason"] = "not_found"
                        return status
                    prev = md["img_dims"].get("height")
                    md["img_dims"]["height"] = int(value)
                    try:
                        self._validate_structure()
                        status["applied"] = True
                        status["stored_value"] = md["img_dims"]["height"]
                    except Exception:
                        md["img_dims"]["height"] = prev
                        status["reason"] = "schema_validation_failed"
                    return status

                # metadata.dpi
                if rest[0] == "dpi":
                    prev = md.get("dpi")
                    md["dpi"] = None if value is None else float(value)
                    try:
                        self._validate_structure()
                        status["applied"] = True
                        status["stored_value"] = md["dpi"]
                    except Exception:
                        md["dpi"] = prev
                        status["reason"] = "schema_validation_failed"
                    return status

                # metadata.color / format / image_name / date_creation (directos)
                leaf = rest[0]
                if leaf in ("color", "format", "image_name", "date_creation"):
                    prev = md.get(leaf)
                    md[leaf] = None if value is None else str(value)
                    try:
                        self._validate_structure()
                        status["applied"] = True
                        status["stored_value"] = md[leaf]
                    except Exception:
                        md[leaf] = prev
                        status["reason"] = "schema_validation_failed"
                    return status

                status["reason"] = "unsupported_metadata_path"
                return status

            # POLYGON
            if root == "polygon":
                if not poly_id:
                    status["reason"] = "poly_id_required"
                    return status
                polys = self.dict_id.get("image_data", {}).get("polygons", {})
                if poly_id not in polys:
                    status["reason"] = "not_found"
                    return status
                poly = polys[poly_id]
                if not isinstance(poly, dict):
                    status["reason"] = "not_found"
                    return status

                # polygon.geometry ...
                if rest[0] == "geometry":
                    if len(rest) == 1:
                        # bloque completo geometry
                        if not isinstance(value, dict):
                            status["reason"] = "invalid_value_type"
                            return status
                        prev = poly.get("geometry")
                        geom_dict = {
                            "polygon_coords": [[float(p[0]), float(p[1])] for p in value.get("polygon_coords", [])],
                            "bounding_box": [float(x) for x in value.get("bounding_box", [])],
                            "centroid": [float(x) for x in value.get("centroid", [])],
                        }
                        poly["geometry"] = geom_dict
                        try:
                            self._validate_structure()
                            status["applied"] = True
                            status["stored_value"] = geom_dict
                        except Exception:
                            poly["geometry"] = prev
                            status["reason"] = "schema_validation_failed"
                        return status

                    # geometry subcampos
                    leaf = rest[1]
                    if "geometry" not in poly or not isinstance(poly["geometry"], dict):
                        status["reason"] = "not_found"
                        return status
                    g = poly["geometry"]

                    if leaf == "polygon_coords":
                        prev = g.get("polygon_coords")
                        try:
                            coords = [[float(p[0]), float(p[1])] for p in value]
                        except Exception:
                            status["reason"] = "invalid_value_type"
                            return status
                        g["polygon_coords"] = coords
                        try:
                            self._validate_structure()
                            status["applied"] = True
                            status["stored_value"] = g["polygon_coords"]
                        except Exception:
                            g["polygon_coords"] = prev
                            status["reason"] = "schema_validation_failed"
                        return status

                    if leaf == "bounding_box":
                        prev = g.get("bounding_box")
                        try:
                            bbox = [float(x) for x in value]
                        except Exception:
                            status["reason"] = "invalid_value_type"
                            return status
                        g["bounding_box"] = bbox
                        try:
                            self._validate_structure()
                            status["applied"] = True
                            status["stored_value"] = g["bounding_box"]
                        except Exception:
                            g["bounding_box"] = prev
                            status["reason"] = "schema_validation_failed"
                        return status

                    if leaf == "centroid":
                        prev = g.get("centroid")
                        try:
                            centroid = [float(value[0]), float(value[1])]
                        except Exception:
                            status["reason"] = "invalid_value_type"
                            return status
                        g["centroid"] = centroid
                        try:
                            self._validate_structure()
                            status["applied"] = True
                            status["stored_value"] = g["centroid"]
                        except Exception:
                            g["centroid"] = prev
                            status["reason"] = "schema_validation_failed"
                        return status

                    status["reason"] = "unsupported_geometry_path"
                    return status

                # polygon.cropedd_geometry ...
                if rest[0] == "cropedd_geometry":
                    if len(rest) == 1:
                        # bloque completo cropedd_geometry
                        if not isinstance(value, dict):
                            status["reason"] = "invalid_value_type"
                            return status
                        prev = poly.get("cropedd_geometry")
                        cg = {
                            "padding_bbox": [float(x) for x in value.get("padding_bbox", [])],
                            "padd_centroid": [float(x) for x in value.get("padd_centroid", [])],
                            "padding_coords": [int(x) for x in value.get("padding_coords", [])],
                            "perimeter": (None if value.get("perimeter") is None else float(value.get("perimeter"))),
                            "was_fragmented": bool(value.get("was_fragmented", False)),
                        }
                        poly["cropedd_geometry"] = cg
                        try:
                            self._validate_structure()
                            status["applied"] = True
                            status["stored_value"] = cg
                        except Exception:
                            poly["cropedd_geometry"] = prev
                            status["reason"] = "schema_validation_failed"
                        return status

                    if "cropedd_geometry" not in poly or not isinstance(poly["cropedd_geometry"], dict):
                        status["reason"] = "not_found"
                        return status
                    cg = poly["cropedd_geometry"]
                    leaf = rest[1]

                    if leaf == "padding_bbox":
                        prev = cg.get("padding_bbox")
                        try:
                            cg["padding_bbox"] = [float(x) for x in value]
                        except Exception:
                            status["reason"] = "invalid_value_type"
                            return status
                        try:
                            self._validate_structure()
                            status["applied"] = True
                            status["stored_value"] = cg["padding_bbox"]
                        except Exception:
                            cg["padding_bbox"] = prev
                            status["reason"] = "schema_validation_failed"
                        return status

                    if leaf == "padd_centroid":
                        prev = cg.get("padd_centroid")
                        try:
                            cg["padd_centroid"] = [float(value[0]), float(value[1])]
                        except Exception:
                            status["reason"] = "invalid_value_type"
                            return status
                        try:
                            self._validate_structure()
                            status["applied"] = True
                            status["stored_value"] = cg["padd_centroid"]
                        except Exception:
                            cg["padd_centroid"] = prev
                            status["reason"] = "schema_validation_failed"
                        return status

                    if leaf == "padding_coords":
                        prev = cg.get("padding_coords")
                        try:
                            cg["padding_coords"] = [int(x) for x in value]
                        except Exception:
                            status["reason"] = "invalid_value_type"
                            return status
                        try:
                            self._validate_structure()
                            status["applied"] = True
                            status["stored_value"] = cg["padding_coords"]
                        except Exception:
                            cg["padding_coords"] = prev
                            status["reason"] = "schema_validation_failed"
                        return status

                    if leaf == "perimeter":
                        prev = cg.get("perimeter")
                        cg["perimeter"] = None if value is None else float(value)
                        try:
                            self._validate_structure()
                            status["applied"] = True
                            status["stored_value"] = cg["perimeter"]
                        except Exception:
                            cg["perimeter"] = prev
                            status["reason"] = "schema_validation_failed"
                        return status

                    if leaf == "was_fragmented":
                        prev = cg.get("was_fragmented")
                        cg["was_fragmented"] = bool(value)
                        try:
                            self._validate_structure()
                            status["applied"] = True
                            status["stored_value"] = cg["was_fragmented"]
                        except Exception:
                            cg["was_fragmented"] = prev
                            status["reason"] = "schema_validation_failed"
                        return status

                    status["reason"] = "unsupported_cropedd_geometry_path"
                    return status

                # polygon.ocr ...
                if rest[0] == "ocr":
                    if len(rest) == 1:
                        # bloque completo ocr
                        if not isinstance(value, dict):
                            status["reason"] = "invalid_value_type"
                            return status
                        prev = poly.get("ocr")
                        conf_raw = value.get("confidence", 60.0)
                        conf = float(conf_raw)
                        if conf < 60.0:
                            conf = 60.0
                            status["sanitized"] = True
                        if conf > 100.0:
                            conf = 100.0
                            status["sanitized"] = True
                        ocr_dict = {
                            "ocr_raw": (None if value.get("ocr_raw") is None else str(value.get("ocr_raw"))),
                            "confidence": conf,
                        }
                        poly["ocr"] = ocr_dict
                        try:
                            self._validate_structure()
                            status["applied"] = True
                            status["stored_value"] = ocr_dict
                        except Exception:
                            poly["ocr"] = prev
                            status["reason"] = "schema_validation_failed"
                        return status

                    if "ocr" not in poly or not isinstance(poly["ocr"], dict):
                        status["reason"] = "not_found"
                        return status
                    ocr = poly["ocr"]
                    leaf = rest[1]

                    if leaf == "ocr_raw":
                        prev = ocr.get("ocr_raw")
                        ocr["ocr_raw"] = None if value is None else str(value)
                        try:
                            self._validate_structure()
                            status["applied"] = True
                            status["stored_value"] = ocr["ocr_raw"]
                        except Exception:
                            ocr["ocr_raw"] = prev
                            status["reason"] = "schema_validation_failed"
                        return status

                    if leaf == "confidence":
                        prev = ocr.get("confidence")
                        try:
                            conf = float(value)
                        except Exception:
                            status["reason"] = "invalid_value_type"
                            return status
                        if conf < 60.0:
                            conf = 60.0
                            status["sanitized"] = True
                        if conf > 100.0:
                            conf = 100.0
                            status["sanitized"] = True
                        ocr["confidence"] = conf
                        try:
                            self._validate_structure()
                            status["applied"] = True
                            status["stored_value"] = ocr["confidence"]
                        except Exception:
                            ocr["confidence"] = prev
                            status["reason"] = "schema_validation_failed"
                        return status

                    status["reason"] = "unsupported_ocr_path"
                    return status

                # polygon.line_id
                if rest[0] == "line_id":
                    prev = poly.get("line_id")
                    poly["line_id"] = str(value)
                    try:
                        self._validate_structure()
                        status["applied"] = True
                        status["stored_value"] = poly["line_id"]
                    except Exception:
                        poly["line_id"] = prev
                        status["reason"] = "schema_validation_failed"
                    return status

                # polygon.cropped_img (np.ndarray por referencia)
                if rest[0] == "cropped_img":
                    prev = poly.get("cropped_img")
                    # No copiamos np.ndarray; asignación directa (in-place policy) [[memory:4316149]]
                    if not isinstance(value, np.ndarray):
                        status["reason"] = "invalid_value_type"
                        return status
                    poly["cropped_img"] = value
                    try:
                        self._validate_structure()
                        status["applied"] = True
                        status["stored_value"] = "<ndarray>"
                    except Exception:
                        poly["cropped_img"] = prev
                        status["reason"] = "schema_validation_failed"
                    return status

                status["reason"] = "unsupported_polygon_path"
                return status

            # LINE
            if root == "line":
                if not line_id:
                    status["reason"] = "line_id_required"
                    return status
                lines = self.dict_id.get("image_data", {}).get("all_lines", {})
                if line_id not in lines:
                    status["reason"] = "not_found"
                    return status
                line = lines[line_id]
                if not isinstance(line, dict):
                    status["reason"] = "not_found"
                    return status

                leaf = rest[0]
                if leaf == "line_bbox":
                    prev = line.get("line_bbox")
                    try:
                        line["line_bbox"] = [float(x) for x in value]
                    except Exception:
                        status["reason"] = "invalid_value_type"
                        return status
                    try:
                        self._validate_structure()
                        status["applied"] = True
                        status["stored_value"] = line["line_bbox"]
                    except Exception:
                        line["line_bbox"] = prev
                        status["reason"] = "schema_validation_failed"
                    return status

                if leaf == "line_centroid":
                    prev = line.get("line_centroid")
                    try:
                        line["line_centroid"] = [float(value[0]), float(value[1])]
                    except Exception:
                        status["reason"] = "invalid_value_type"
                        return status
                    try:
                        self._validate_structure()
                        status["applied"] = True
                        status["stored_value"] = line["line_centroid"]
                    except Exception:
                        line["line_centroid"] = prev
                        status["reason"] = "schema_validation_failed"
                    return status

                if leaf == "polygon_ids":
                    prev = line.get("polygon_ids")
                    try:
                        line["polygon_ids"] = [str(x) for x in value]
                    except Exception:
                        status["reason"] = "invalid_value_type"
                        return status
                    try:
                        self._validate_structure()
                        status["applied"] = True
                        status["stored_value"] = line["polygon_ids"]
                    except Exception:
                        line["polygon_ids"] = prev
                        status["reason"] = "schema_validation_failed"
                    return status

                status["reason"] = "unsupported_line_path"
                return status

            status["reason"] = "unsupported_root"
            return status

        except Exception as e:
            # Nunca explota hacia afuera
            status["success"] = False
            status["applied"] = False
            status["reason"] = f"internal_error: {e}"
            return status

    def set_polygon_geometry(self, poly_id: str, geometry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Atajo: inserta bloque completo de geometry (sin crear si no existe).
        """
        return self.update_data("polygon.geometry", geometry, poly_id=poly_id)

    def set_polygon_cropedd_geometry(self, poly_id: str, crop: Dict[str, Any]) -> Dict[str, Any]:
        """
        Atajo: inserta bloque completo de cropedd_geometry (sin crear si no existe).
        """
        return self.update_data("polygon.cropedd_geometry", crop, poly_id=poly_id)

    def set_polygon_ocr(self, poly_id: str, ocr_text: Optional[str], confidence: float) -> Dict[str, Any]:
        """
        Atajo: inserta bloque completo de ocr. Confianza se clamp a [60, 100].
        """
        return self.update_data("polygon.ocr", {"ocr_raw": ocr_text, "confidence": confidence}, poly_id=poly_id)

    def set_polygon_line_id(self, poly_id: str, line_id: str) -> Dict[str, Any]:
        """
        Atajo: asigna line_id del polígono (sin crear si no existe).
        """
        return self.update_data("polygon.line_id", line_id, poly_id=poly_id)

    def set_polygon_cropped_img(self, poly_id: str, cropped_img: np.ndarray) -> Dict[str, Any]:
        """
        Atajo: asigna cropped_img por referencia (np.ndarray intocable, sin copias).
        """
        return self.update_data("polygon.cropped_img", cropped_img, poly_id=poly_id)


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
}

