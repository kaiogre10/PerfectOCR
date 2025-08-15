# core/domain/workflow_manager.py
from core.domain.data_models import WORKFLOW_SCHEMA, WorkflowDict
import numpy as np
import jsonschema
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

class DataFormatter:
    """
    Válvula de entrada/salida para todas las operaciones del dict.
    Los workers NO tocan directamente el dict_id, solo pasan por aquí.
    """

    def __init__(self):
        self.workflow: Optional[WorkflowDict] = None
        self.schema = WORKFLOW_SCHEMA

    def create_dict(self, dict_id: str, full_img: np.ndarray[Any, Any], metadata: Dict[str, Any]) -> bool:
        """Crea un nuevo dict"""

        self.workflow_dict: Dict[str, Any] = {
            "dict_id": dict_id,
            "full_img": [full_img.tolist() if hasattr(full_img, 'tolist') else full_img],
            "metadata": {
                "image_name": str(metadata.get("image_name", "")),
                "format": str(metadata.get("format", "")),
                "img_dims": {
                    "width": int(metadata.get("img_dims", {}).get("width")),
                    "height": int(metadata.get("img_dims", {}).get("height")),
                },
                "dpi": (
                    metadata.get("dpi") if isinstance(metadata.get("dpi"), dict)
                    else {"x": float(metadata.get("dpi", 0)), "y": None}
                ),
                "date_creation": metadata.get("date_creation", datetime.now().isoformat()),
                "color": str(metadata.get("color", "")) if metadata.get("color") is not None else None
            },
            "polygons": {}
        }

        try:
            jsonschema.validate(self.workflow_dict, self.schema)
            return True
        except Exception as e:
            logger.error(f"Error validando workflow_dict: {e}")
            return False
        
    def create_polygon_dicts(self, results: Optional[List[Any]]) -> bool:
        """
        Procesa los resultados de PaddleOCR y los guarda como diccionario de polígonos en el workflow_dict.
        """
        
        polygons: Dict[str, Dict[str, Any]] = {}
        try:
            for idx, poly_pts in enumerate(results[0]):
                xs = [float(p[0]) for p in poly_pts]
                ys = [float(p[1]) for p in poly_pts]
                poly_id = f"poly_{idx:04d}"
                polygons[poly_id] = {
                    "polygon_id": poly_id,
                    "geometry": {
                        "polygon_coords": [[xs[i], ys[i]] for i in range(len(xs))],
                        "bounding_box": [min(xs), min(ys), max(xs), max(ys)],
                        "centroid": [sum(xs) / len(xs), sum(ys) / len(ys)]
                    },
                    "cropped_geometry": {
                        "padding_bbox": [],
                        "padd_centroid": [],
                        "padding_coords": [],
                    },
                    "cropped_img": None,
                    "perimeter": None,
                    "line_id": "",
                    "ocr_text": "",
                    "ocr_confidence": None,
                    "was_fragmented": False,
                    "status": False,
                    "stage": ""
                }
            if self.workflow_dict:
                self.workflow_dict["polygons"] = polygons
                logger.info(f"Polígonos estructurados: {len(polygons)}")
                return True
            else:
                logger.error("No hay workflow_dict inicializado.")
                return False
        except Exception as e:
            logger.error(f"Error creando diccionario de polígonos: {e}")
            return False

    def get_dict_data(self) -> Dict[str, Any]:
        """Devuelve copia completa del dict"""
        return self.workflow_dict if self.workflow_dict else {}
    
    def get_metadata(self) -> Dict[str, Any]:
        """Devuelve los metadatos del dict"""
        return self.workflow_dict["metadata"] if self.workflow_dict else {}

    def get_polygons(self) -> Dict[str, Any]:
        return self.workflow_dict["polygons"] if self.workflow_dict else {}
        
    def get_polygons_with_cropped_img(self) -> Dict[str, Dict[str, Any]]:
        """
        Devuelve el diccionario de polígonos con sus imágenes recortadas listas para el contexto de los workers.
        """
        if not self.workflow_dict:
            return {}
        return self.workflow_dict["polygons"]

    def update_full_img(self, dict_id: str, full_img: (Optional[np.ndarray[Any, Any]])=None) -> bool:
        """Actualiza o vacía la imagen completa en el workflow_dict"""
        try:
            if not self.workflow_dict:
                logger.error("No hay workflow_dict inicializado para actualizar full_img.")
                return False
                
            if full_img is None:
                # Si se pasa None, vaciamos la imagen para liberar memoria
                self.workflow_dict["full_img"] = None
                logger.info("full_img liberada del workflow_dict.")
                return True
            else:
                # Si se pasa una imagen, la actualizamos
                self.workflow_dict["full_img"] = full_img.tolist()
                logger.info("full_img actualizada en el workflow_dict.")
            return True
        except Exception as e:
            logger.error(f"Error actualizando full_img: {e}")
            return False
            
    def save_cropped_images(
        self,
        cropped_images: Dict[str, np.ndarray[Any, Any]],
        line_ids: Dict[str, str],
        cropped_geometries: Dict[str, Dict[str, Any]]
    ) -> bool:
        """Guarda imágenes recortadas, line_ids y geometría de recorte en los polígonos del workflow_dict"""
        try:
            if not self.workflow_dict:
                logger.error("No hay workflow_dict inicializado para guardar imágenes recortadas.")
                return False

            for poly_id, img in cropped_images.items():
                if poly_id in self.workflow_dict["polygons"]:
                    self.workflow_dict["polygons"][poly_id]["cropped_img"] = img.tolist()
                    if poly_id in cropped_geometries:
                        self.workflow_dict["polygons"][poly_id]["cropped_geometry"] = cropped_geometries[poly_id]

            for poly_id, line_id in line_ids.items():
                if poly_id in self.workflow_dict["polygons"]:
                    self.workflow_dict["polygons"][poly_id]["line_id"] = line_id

            logger.info(f"Guardadas {len(cropped_images)} imágenes recortadas, {len(line_ids)} line_ids y geometría de recorte.")
            return True
        except Exception as e:
            logger.error(f"Error guardando imágenes recortadas y geometría: {e}")
            return False
            

        
    def get_cropped_images_for_preprocessing(self) -> Dict[str, np.ndarray[Any, Any]]:
        """
        Devuelve un diccionario de imágenes recortadas listas para preprocesamiento.
        cropped_images = {
            "poly_0000": np.ndarray,  # Imagen numpy del polígono
            "poly_0001": np.ndarray,  # Imagen numpy del polígono
            ...
        }
        """
        cropped_images: Dict[str, np.ndarray[Any, Any]] = {}
        if not self.workflow_dict or not self.workflow_dict:
            return cropped_images

        for poly_id, poly_data in self.workflow_dict["polygons"].items():
            cropped_img = poly_data.get("cropped_img")
            if cropped_img is not None:
                # Si la imagen está en formato lista, conviértela a np.ndarray
                if isinstance(cropped_img, list):
                    cropped_img = np.array(cropped_img, np.uint8)
                cropped_images[poly_id] = cropped_img
        return cropped_images
        
    # NUEVO método en DataFormatter  
    def update_preprocessing_result(self, poly_id: str, cropped_img: np.ndarray[Any, Any], 
                                worker_name: str, success: bool):
        """Actualiza resultado de preprocesamiento y marca stage/status"""
        if poly_id in self.workflow_dict["polygons"]:
            # Actualizar imagen
            self.workflow_dict["polygons"][poly_id]["cropped_img"] = cropped_img
            # Actualizar metadatos
            self.workflow_dict["polygons"][poly_id]["stage"] = worker_name
            self.workflow_dict["polygons"][poly_id]["status"] = success
