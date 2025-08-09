# core/domain/workflow_manager.py
import jsonschema
import numpy as np
from typing import Dict, Any, List
from datetime import datetime
import logging
from core.domain.workflow_models import WORKFLOW_SCHEMA, data_paths

logger = logging.getLogger(__name__)

class DataFormatter:
    """
    Válvula de entrada/salida para todas las operaciones del dict.
    Los workers NO tocan directamente el dict_id, solo pasan por aquí.
    """

    def __init__(self):
        self.dict_id: Dict[str, Any] = {}
        self.schema = WORKFLOW_SCHEMA

    def create_dict(self, dict_id: str, full_img: np.ndarray, metadata: Dict[str, Any]) -> bool: # type: ignore
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
            logger.error(f"Error creando dict: {e}")
            return False

    def update_data(self, abstract: Dict[str, Any]) -> bool:
        """
        Función de entrada universal para actualizar el dict desde un contexto de worker.
        Busca el path usando el alias y escribe los valores contrastando por ID.
        """

        try:
            for alias, data_by_id in abstract.items():
                if alias not in data_paths:
                    logger.warning(f"Alias '{alias}' no encontrado en data_paths. Omitiendo.")
                    continue

                path_template = data_paths[alias]
                
                if not isinstance(data_by_id, dict):
                    logger.warning(f"Datos para el alias '{alias}' no es un diccionario de IDs. Omitiendo.")
                    continue

                for entity_id, value in data_by_id.items():
                    self._set_value_at_path(path_template, entity_id, alias, value)
            
            self._validate_structure()
            return True
        except Exception as e:
            logger.error(f"Error actualizando datos desde contexto: {e}", exc_info=True)
            return False

    def _set_value_at_path(self, path_template: List[str], entity_id: str, leaf_key: str, value: Any):
        """
        Navega o crea la ruta en el dict y establece el valor en la hoja final.
        La ruta en data_paths viene invertida (hoja -> raíz), así que la procesamos al revés.
        """
        current_level = self.dict_id.setdefault("image_data", {})
        
        # Invertimos la plantilla para navegar desde la raíz ('polygons') hacia la hoja ('geometry')
        for key in reversed(path_template):
            # Reemplazamos el placeholder por el ID real
            if key in ["{poly_id}", "{line_id}"]:
                actual_key = entity_id
            else:
                actual_key = key
            
            current_level = current_level.setdefault(actual_key, {})
        
        # Coerción y estructuración intermedia basada en el alias
        if leaf_key == 'confidence':
            value = max(60.0, min(100.0, float(value)))
        elif leaf_key == 'was_fragmented':
            value = bool(value)
        
        # Escribir el valor en la hoja
        current_level[leaf_key] = value

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
