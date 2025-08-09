# core/domain/workflow_dict.py

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
