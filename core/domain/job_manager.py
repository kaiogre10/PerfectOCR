# core/domain/job_manager.py
import jsonschema
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime

class JobManager:
    """
    Válvula de entrada/salida para todas las operaciones del job.
    Los workers NO tocan directamente el job_data, solo pasan por aquí.
    """
    
    def __init__(self):
        self.job_data: Dict[str, Any] = {}
        self._schema = WORKFLOW_SCHEMA
    
    # === MÉTODOS DE ENTRADA (Workers -> JobManager) ===
    
    def create_job(self, job_id: str, full_img: np.ndarray, metadata: Dict[str, Any]) -> bool:
        """Crea un nuevo job con validación automática"""
        try:
            self.job_data = {
                "job_id": job_id,
                "full_img": full_img,
                "document_dict": {
                    "metadata": self._validate_metadata(metadata),
                    "polygons": {},
                    "all_lines": {}
                }
            }
            self._validate_structure()
            return True
        except Exception as e:
            print(f"❌ Error creando job: {e}")
            return False
    
    def add_polygon(self, polygon_data: Dict[str, Any]) -> bool:
        """Agrega polígono con validación automática"""
        try:
            # Transformar datos simples a estructura completa
            validated_polygon = self._transform_polygon(polygon_data)
            
            # Agregar al job
            poly_id = validated_polygon["polygon_id"]
            self.job_data["document_dict"]["polygons"][poly_id] = validated_polygon
            
            # Validar estructura completa
            self._validate_structure()
            return True
        except Exception as e:
            print(f"❌ Error agregando polígono: {e}")
            return False
    
    def add_line(self, line_data: Dict[str, Any]) -> bool:
        """Agrega línea con validación automática"""
        try:
            validated_line = self._transform_line(line_data)
            line_id = validated_line["line_id"]
            self.job_data["document_dict"]["all_lines"][line_id] = validated_line
            self._validate_structure()
            return True
        except Exception as e:
            print(f"❌ Error agregando línea: {e}")
            return False
    
    def update_polygon_ocr(self, poly_id: str, ocr_text: str, confidence: float) -> bool:
        """Actualiza OCR de un polígono específico"""
        try:
            if poly_id not in self.job_data["document_dict"]["polygons"]:
                raise ValueError(f"Polígono {poly_id} no existe")
            
            self.job_data["document_dict"]["polygons"][poly_id]["ocr"] = {
                "ocr_raw": str(ocr_text),
                "confidence": float(confidence)
            }
            self._validate_structure()
            return True
        except Exception as e:
            print(f"Error actualizando OCR: {e}")
    
    def get_job_data(self) -> Dict[str, Any]:
        """Devuelve copia completa del job"""
        return self.job_data.copy()
    
    def get_polygons(self) -> Dict[str, Any]:
        """Devuelve solo los polígonos"""
        return self.job_data["document_dict"]["polygons"].copy()
    
    def get_polygon(self, poly_id: str) -> Optional[Dict[str, Any]]:
        """Devuelve un polígono específico"""
        return self.job_data["document_dict"]["polygons"].get(poly_id)
    
    def get_lines(self) -> Dict[str, Any]:
        """Devuelve solo las líneas"""
        return self.job_data["document_dict"]["all_lines"].copy()
    
    def get_metadata(self) -> Dict[str, Any]:
        """Devuelve solo los metadatos"""
        return self.job_data["document_dict"]["metadata"].copy()
    
    # === MÉTODOS INTERNOS DE VALIDACIÓN ===
    
    def _validate_structure(self) -> None:
        """Valida estructura completa contra schema"""
        jsonschema.validate(self.job_data, self._schema)
    
    def _validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Transforma y valida metadatos"""
        return {
            "doc_name": str(metadata.get("doc_name", "")),
            "formato": str(metadata.get("formato", "")),
            "img_dims": {
                "width": int(metadata.get("img_dims", {}).get("width", 0)),
                "height": int(metadata.get("img_dims", {}).get("height", 0))
            },
            "dpi": float(metadata["dpi"]) if metadata.get("dpi") is not None else None,
            "date": metadata.get("date", datetime.now().isoformat()),
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
            "bounding_box": [float(x) for x in line_data["bounding_box"]],
            "centroid": [float(x) for x in line_data["centroid"]],
            "polygon_ids": [str(x) for x in line_data["polygon_ids"]]
        }