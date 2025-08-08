# core/domain/workflow_schema.py
import jsonschema
from typing import Dict, Any

WORKFLOW_SCHEMA = {
    "type": "object",
    "properties": {
        "job_id": {"type": "string"},
        "full_img": {"type": "object"},  # np.ndarray real
        "document_dict": {
            "type": "object",
            "properties": {
                "metadata": {
                    "type": "object",
                    "properties": {
                        "doc_name": {"type": "string"},
                        "formato": {"type": "string"},
                        "img_dims": {
                            "type": "object",
                            "properties": {
                                "width": {"type": "integer"},
                                "height": {"type": "integer"}
                            },
                            "required": ["width", "height"]
                        },
                        "dpi": {"type": ["number", "null"]},
                        "date": {"type": "string"},
                        "color": {"type": ["string", "null"]}
                    },
                    "required": ["doc_name", "formato", "img_dims"]
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
                                        "width": {"type": "number"},
                                        "height": {"type": "number"},
                                        "padding_coords": {
                                            "type": "array",
                                            "items": {"type": "integer"},
                                            "minItems": 4,
                                            "maxItems": 4
                                        },
                                        "perimeter": {"type": "number"}
                                    },
                                    "required": ["polygon_coords", "bounding_box", "centroid", "width", "height"]
                                },
                                "line_id": {"type": ["string", "null"]},
                                "cropped_img": {"type": ["object", "null"]},
                                "was_fragmented": {"type": "boolean"},
                                "ocr": {
                                    "type": "object",
                                    "properties": {
                                        "ocr_raw": {"type": "string"},
                                        "confidence": {"type": "number", "minimum": 60, "maximum": 100}
                                    },
                                    "required": ["ocr_raw", "confidence"]
                                },
                            },
                            "required": ["polygon_id", "geometry"]
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
                                "line_bounding_box": {
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
                                },
                            },
                            "required": ["line_bounding_box", "bounding_box", "line_centroid", "polygon_ids"]
                        },
                    },
                },
            },
            "required": ["metadata", "polygons", "all_lines"]
        },
    },
    "required": ["job_id", "full_img", "document_dict"]
}

def validate_job_data(job_data: Dict[str, Any]) -> bool:
    """Valida la estructura del job usando el schema centralizado"""
    try:
        jsonschema.validate(job_data, WORKFLOW_SCHEMA)
        return True
    except jsonschema.ValidationError as e:
        print(f"❌ Error de validación: {e}")
        return False
    
# {"type": "string"}      # Debe ser texto
# {"type": "integer"}     # Debe ser número entero
# {"type": "number"}      # Debe ser número (puede ser decimal)
# {"type": "boolean"}     # Debe ser True/False
# {"type": "object"}      # Debe ser diccionario
# {"type": "array"}       # Debe ser lista