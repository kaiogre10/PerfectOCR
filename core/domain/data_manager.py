# core/domain/workflow_manager.py
import jsonschema
import numpy as np
from typing import Dict, Any, List
from datetime import datetime
import logging
from core.domain.data_models import WORKFLOW_SCHEMA, data_paths

logger = logging.getLogger(__name__)

class DataFormatter:
    """
    Válvula de entrada/salida para todas las operaciones del dict.
    Los workers NO tocan directamente el dict_id, solo pasan por aquí.
    """

    def __init__(self):
        self.dict_id: Dict[str, Any] = {}
        self.schema = WORKFLOW_SCHEMA

    def create_dict(self, dict_id: str, full_img: np.ndarray[Any, Any], metadata: Dict[str, Any]) -> bool: # type: ignore
        """Crea un nuevo dict con validación automática"""
        try:
            self.dict_id = {
                "dict_id": dict_id,
                "full_img": full_img.tolist(),
                "metadata": self._validate_metadata(metadata),
                "image_data": {
                    "polygons": {},
                }
            }
            self._validate_structure()
            return True
        except Exception as e:
            logger.error(f"Error creando dict: {e}")
            return False

    def _update_full_img(self, dict_id: str, full_img: np.ndarray[Any, Any]) -> bool:
        try:
            # Verifica que el dict_id coincida
            if self.dict_id.get("dict_id") != dict_id:
                logger.warning(f"dict_id '{dict_id}' no coincide con el actual.")
                return False
            self.dict_id["full_img"] = full_img.tolist(),
            # Valida la estructura
            self._validate_structure()
            return True
        except Exception as e:
            logger.error(f"Error actualizando full_img: {e}")
            return False
            
    def _set_value_at_path(self, path_template: List[str], entity_id: str, leaf_key: str, value: Any):
        """
        Esta función navega (o crea si no existe) la ruta especificada en el diccionario interno,
        y establece el valor en la hoja final. La ruta se define en data_paths y viene invertida
        (de la hoja a la raíz), por lo que se recorre al revés para construir la estructura desde la raíz.
        """
        # 1. Empieza desde el nivel 'image_data' del diccionario principal.
        current_level = self.dict_id.setdefault("image_data", {})
        
        # 2. Recorre la plantilla de ruta al revés (de raíz a hoja).
        for key in reversed(path_template):
            # 3. Si el key es un placeholder de ID, lo reemplaza por el ID real de la entidad.
            if key in ["{poly_id}", "{line_id}"]:
                actual_key = entity_id
            else:
                actual_key = key
            
            # 4. Baja un nivel en el diccionario, creando el subdiccionario si no existe.
            current_level = current_level.setdefault(actual_key, {})
        
        # 5. Realiza coerción de tipo si el alias lo requiere (por ejemplo, 'confidence' o 'was_fragmented').
        if leaf_key == 'confidence':
            value = max(60.0, min(100.0, float(value)))
        elif leaf_key == 'was_fragmented':
            value = bool(value)
        
        # 6. Finalmente, escribe el valor en la hoja correspondiente.
        current_level[leaf_key] = value

    def _validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Transforma y valida metadatos"""
        return {
            "image_name": str(metadata.get("image_name", "")),
            "format": str(metadata.get("format", "")),
            "img_dims": {
                "width": int(metadata.get("img_dims", {}).get("width")),
                "height": int(metadata.get("img_dims", {}).get("height"))
            },
            "dpi": float(metadata["dpi"]) if metadata.get("dpi") is not None else None,
            "date_creation": metadata.get("date_creation", datetime.now().isoformat()),
            "color": str(metadata["color"]) if metadata.get("color") is not None else None
        }

    def _get_dict_data(self) -> Dict[str, Any]:
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

    # Consulta puntual
    def get_value(self, alias: str, entity_id: str) -> Any:
        if alias not in data_paths:
            return None
        path_template: List[str] = data_paths[alias]
        level: Any = self.dict_id.get("image_data", {})
        for key in reversed(path_template):
            key = entity_id if key in ("{poly_id}", "{line_id}") else key
            level = level.get(key, {})
        return level.get(alias)

    # Actualización puntual
    def set_value(self, alias: str, entity_id: str, value: Any) -> bool:
        try:
            if alias not in data_paths:
                logger.warning(f"Alias desconocido: {alias}")
                return False
            path_template: List[str] = data_paths[alias]
            level: Dict[str, Any] = self.dict_id.setdefault("image_data", {})
            for key in reversed(path_template):
                key = entity_id if key in ("{poly_id}", "{line_id}") else key
                level = level.setdefault(key, {})
            # coerciones simples de ejemplo
            if alias == "confidence":
                value = max(60.0, min(100.0, float(value)))
            if alias == "was_fragmented":
                value = bool(value)
            level[alias] = value
            return True
        except Exception as e:
            logger.error(f"Error set_value({alias}, {entity_id}): {e}", exc_info=True)
            return False

    # Actualización batch (abstract)
    def update_data(self, abstract: Dict[str, Dict[str, Any]]) -> bool:
        try:
            for alias, data_by_id in abstract.items():
                if alias not in data_paths or not isinstance(data_by_id, List):
                    continue
                for entity_id, entity_data in data_by_id.items():
                    # Reutiliza set_value cuando la hoja es un valor:
                    if not isinstance(entity_data, dict):
                        if not self.set_value(alias, str(entity_id), entity_data):
                            return False
                        continue
                    # Si viene un sub-dict (p. ej., varias hojas de un poly), escribe cada clave
                    for sub_alias, sub_value in entity_data.items():
                        if sub_alias not in data_paths:
                            # ruta directa al nodo del poly si se requiere
                            pass
                        else:
                            if not self.set_value(sub_alias, str(entity_id), sub_value):
                                return False
            # Validación opcional por checkpoint:
            jsonschema.validate(self.dict_id, self.schema)
            return True
        except Exception as e:
            logger.error(f"Error update_data: {e}", exc_info=True)
            return False

    def get_dict_data(self) -> Dict[str, Any]:
        return self.dict_id.copy()
