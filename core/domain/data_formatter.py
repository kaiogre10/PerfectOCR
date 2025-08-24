# core/domain/data_formatter.py
from core.domain.data_models import WORKFLOW_SCHEMA, WorkflowDict, DENSITY_ENCODER, StructuredTable, Geometry, Metadata, Polygons, CroppedGeometry, CroppedImage
from dataclasses import asdict
import numpy as np
import jsonschema
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

class DataFormatter:
    """
    Válvula de entrada/salida para todas las operaciones del dict.
    Los workers NO tocan directamente el dict_id, solo pasan por aquí.
    """
    def __init__(self):
        self.workflow: Optional[WorkflowDict] = None
        self.schema = WORKFLOW_SCHEMA
        self.encoder = DENSITY_ENCODER
        self.structured_table: Optional[StructuredTable] = None

    def create_dict(self, dict_id: str, full_img: np.ndarray[Any, np.dtype[np.uint8]], metadata: Dict[str, Any]) -> bool:
        """Crea un nuevo dict"""
        self.temp_dict: Dict[str, Any] = {
            "dict_id": dict_id,
            "full_img": [full_img.tolist() if hasattr(full_img, 'tolist') else full_img],
            "metadata": {
                "image_name": str(metadata.get("image_name", "")),
                "format": str(metadata.get("format", "")),
                "img_dims": {
                    "width": int(metadata.get("img_dims", {}).get("width")),
                    "height": int(metadata.get("img_dims", {}).get("height")),
                    "size": int(metadata.get("img_dims", {}).get("size")),
                },
                "dpi": (
                    metadata.get("dpi") if isinstance(metadata.get("dpi"), dict)
                    else {"x": float(metadata.get("dpi", 0)), "y": None}
                ),
                "date_creation": metadata.get("date_creation", datetime.now().isoformat()),
                "color": str(metadata.get("color", "")) if metadata.get("color") is not None else None
            },
            "polygons": {},
            "all_lines": {},
            "tabular_lines": {}
        }

        try:
            jsonschema.validate(self.temp_dict, self.schema)
            metadata_obj = Metadata(
                image_name=self.temp_dict["metadata"]["image_name"],
                format=self.temp_dict["metadata"]["format"],
                img_dims=self.temp_dict["metadata"]["img_dims"],
                dpi=self.temp_dict["metadata"]["dpi"],
                date_creation=self.temp_dict["metadata"]["date_creation"],
                color=self.temp_dict["metadata"]["color"]
            )

            self.workflow = WorkflowDict(
                dict_id=dict_id,
                full_img=full_img,
                metadata=metadata_obj,
                polygons={},
                all_lines={},
                tabular_lines={}
            )

            self.workflow_dict = self.temp_dict

            return True
        except Exception as e:
            logger.error(f"Error validando workflow_dict: {e}")
            return False
        
    def _validate_and_create_polygon(self, poly_data: Dict[str, Any]) -> Optional[Polygons]:
        """Valida un polígono individual contra el schema y crea la dataclass"""
        try:
            # Crear esquema temporal para un polígono
            poly_schema = self.schema["properties"]["polygons"]["patternProperties"]["^poly_\\d{4}$"]
            
            # Validar estructura
            jsonschema.validate(poly_data, poly_schema)
            
            # Crear dataclasses anidadas
            geometry = Geometry(
                polygon_coords=poly_data["geometry"]["polygon_coords"],
                bounding_box=poly_data["geometry"]["bounding_box"],
                centroid=poly_data["geometry"]["centroid"]
            )
            
            cropped_geo = CroppedGeometry(
                padd_centroid=poly_data["cropped_geometry"]["padd_centroid"],
                padding_coords=poly_data["cropped_geometry"]["padding_coords"],
                poly_dims=poly_data["cropped_geometry"].get("poly_dims", {})
            )
            
            # Crear polígono completo
            polygon = Polygons(
                polygon_id=poly_data["polygon_id"],
                geometry=geometry,
                cropedd_geometry=cropped_geo,
                cropped_img=CroppedImage(poly_data["cropped_img"]) if poly_data["cropped_img"] else None,
                perimeter=poly_data.get("perimeter"),
                line_id=poly_data.get("line_id", ""),
                ocr_text=poly_data.get("ocr_text"),
                ocr_confidence=poly_data.get("ocr_confidence"),
                was_fragmented=poly_data.get("was_fragmented", False),
                status=poly_data.get("status", True),
                stage=poly_data.get("stage", "")
            )
            
            return polygon
        except jsonschema.ValidationError as e:
            logger.error(f"Polígono no válido: {e}")
            return None
        except Exception as e:
            logger.error(f"Error creando polígono: {e}")
            return None

    def create_polygon_dicts(self, results: Optional[List[Any]]) -> bool:
        """Refactorizado para usar validación + dataclasses"""
        try:
            if results is None:
                return False
            
            polygons_dict: Dict[str, Dict[str, Any]] = {}
            polygons_dataclass: Dict[str, Polygons] = {}
            
            for idx, poly_pts in enumerate(results[0]):
                poly_id = f"poly_{idx:04d}"
                
                # Cálculos vectorizados (igual que antes)
                coords = np.array([[float(p[0]), float(p[1])] for p in poly_pts])
                bbox = np.array([coords[:, 0].min(), coords[:, 1].min(), 
                            coords[:, 0].max(), coords[:, 1].max()])
                centroid = coords.mean(axis=0)
                
                # Crear estructura para validación
                poly_data: Dict[str, Any] = {
                    "polygon_id": poly_id,
                    "geometry": {
                        "polygon_coords": coords.tolist(),
                        "bounding_box": bbox.tolist(),
                        "centroid": centroid.tolist()
                    },
                    "cropped_geometry": {
                        "padd_centroid": [],
                        "padding_coords": [],
                        "poly_dims": {}
                    },
                    "cropped_img": None,
                    "perimeter": None,
                    "line_id": "",
                    "ocr_text": "",
                    "ocr_confidence": None,
                    "was_fragmented": False,
                    "status": True,
                    "stage": ""
                }
                
                # Validar y crear dataclass
                polygon_obj = self._validate_and_create_polygon(poly_data)
                if polygon_obj:
                    polygons_dict[poly_id] = poly_data
                    polygons_dataclass[poly_id] = polygon_obj
            
            # Actualizar ambas estructuras
            if self.workflow_dict:
                self.workflow_dict["polygons"] = polygons_dict
            if self.workflow:
                self.workflow.polygons = polygons_dataclass
                
            logger.debug(f"Polígonos creados y validados: {len(polygons_dict)}")
            return True
            
        except Exception as e:
            logger.error(f"Error en create_polygon_dicts: {e}")
            return False

    def get_dict_data(self) -> Dict[str, Any]:
        """Devuelve copia completa del dict"""
        return self.workflow_dict if self.workflow_dict else {}
    
    def get_metadata(self) -> Dict[str, Any]:
        """Devuelve los metadatos del dict"""
        return self.workflow_dict["metadata"] if self.workflow_dict else {}

    def get_polygons(self) -> Dict[str, Polygons]:
        return self.workflow.polygons if self.workflow else {}
        
    def get_tabular_lines(self) -> Dict[str, Any]:
        return self.workflow_dict["tabular_lines"] if self.workflow_dict else {}
        
    def get_structured_table(self) -> Optional[pd.DataFrame]:
        return self.structured_table.df if self.structured_table else None

    def get_structured_semantic_types(self) -> Optional[List[str]]:
        return self.structured_table.semantic_types if self.structured_table else None

    def get_all_lines(self) -> Dict[str, Any]:
        return self.workflow_dict["all_lines"] if self.workflow_dict else {}
        
    def get_polygons_with_cropped_img(self) -> Dict[str, Dict[str, Any]]:
        """
        Devuelve el diccionario de polígonos con sus imágenes recortadas listas para el contexto de los workers.
        """
        if not self.workflow_dict:
            return {}
        return self.workflow_dict["polygons"]
        
    def get_encode_lines(self, line_ids: Optional[List[str]] = None) -> Dict[str, List[int]]:
        """
        Codifica líneas específicas usando DENSITY_ENCODER con operaciones optimizadas.
        Si no se especifican line_ids, codifica todas las líneas existentes.
        """
        try:
            if not self.workflow_dict or "all_lines" not in self.workflow_dict:
                logger.warning("No hay líneas disponibles para codificar.")
                return {}
                
            encoded_lines: Dict[str, List[int]] = {}
            all_lines = self.workflow_dict["all_lines"]
            lines_to_encode = line_ids if line_ids is not None else list(all_lines.keys())
            
            # Codificación optimizada para todas las líneas
            for line_id in lines_to_encode:
                if line_id in all_lines:
                    line_text = all_lines[line_id].get("text", "")
                    if line_text:
                        compact_text = ''.join(line_text.split())
                        # Usar list comprehension para codificación más eficiente
                        encoded_text = [self.encoder.get(char, 0) for char in compact_text]
                        encoded_lines[line_id] = encoded_text
                        all_lines[line_id]["encoded_text"] = encoded_text
                    else:
                        logger.warning(f"Línea {line_id} no tiene texto para codificar.")
                else:
                    logger.warning(f"Línea {line_id} no encontrada en all_lines.")
                
            logger.debug(f"Codificadas {len(encoded_lines)} líneas para análisis de densidad.")
            return encoded_lines
        except Exception as e:
            logger.error(f"Error codificando líneas: {e}")
            return {}

    def update_full_img(self, full_img: (Optional[np.ndarray[Any, np.dtype[np.uint8]]])=None) -> bool:
        """Actualiza o vacía la imagen completa en el workflow_dict"""
        try:
            if not self.workflow_dict:
                logger.error("No hay workflow_dict inicializado para actualizar full_img.")
                return False
                
            if full_img is None:
                # Si se pasa None, vaciamos la imagen para liberar memoria
                self.workflow_dict["full_img"] = None
                logger.debug("full_img liberada del workflow_dict.")
                return True
            else:
                # Si se pasa una imagen, la actualizamos
                self.workflow_dict["full_img"] = full_img.tolist()
                logger.debug("full_img actualizada en el workflow_dict.")
            return True
        except Exception as e:
            logger.error(f"Error actualizando full_img: {e}")
            return False
            
    def save_cropped_images(
        self,
        cropped_images: Dict[str, np.ndarray[Any, np.dtype[np.uint8]]],
        cropped_geometries: Dict[str, Dict[str, Any]]
    ) -> bool:
        """Guarda imágenes recortadas, line_ids y geometría de recorte en los polígonos del workflow_dict"""
        try:
            if not self.workflow_dict:
                logger.error("No hay workflow_dict inicializado para guardar imágenes recortadas.")
                return False

            for poly_id, img in cropped_images.items():
                if poly_id in self.workflow_dict["polygons"]:
                    self.workflow_dict["polygons"][poly_id]["cropped_img"] = img
                    if poly_id in cropped_geometries:
                        self.workflow_dict["polygons"][poly_id]["cropped_geometry"] = cropped_geometries[poly_id]

            logger.debug(f"Guardadas {len(cropped_images)} imágenes recortadas y geometría de recorte.")
            return True
        except Exception as e:
            logger.error(f"Error guardando imágenes recortadas y geometría: {e}")
            return False
             
    def get_cropped_images_for_preprocessing(self) -> Dict[str, np.ndarray[Any, np.dtype[np.uint8]]]:
        """
        Devuelve un diccionario de imágenes recortadas listas para preprocesamiento.
        cropped_images = {
            "poly_0000": np.ndarray,  # Imagen numpy del polígono
            "poly_0001": np.ndarray,  # Imagen numpy del polígono
            ...
        }
        """
        cropped_images: Dict[str, np.ndarray[Any, np.dtype[np.uint8]]] = {}
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
        
    def update_preprocessing_result(self, poly_id: str, cropped_img: np.ndarray[Any, np.dtype[np.uint8]], 
                                worker_name: str, success: bool):
        """Actualiza resultado de preprocesamiento y marca stage/status"""
        if poly_id in self.workflow_dict["polygons"]:
            # Actualizar imagen
            self.workflow_dict["polygons"][poly_id]["cropped_img"] = cropped_img
            # Actualizar metadatos
            self.workflow_dict["polygons"][poly_id]["stage"] = worker_name
            self.workflow_dict["polygons"][poly_id]["status"] = success
            
    def update_ocr_results(self, final_results: List[Optional[Dict[str, Any]]], polygon_ids: List[str]) -> bool:
        """
        Actualiza los resultados de OCR en los polígonos del workflow_dict.
        batch_results: lista de dicts con 'text' y 'confidence'
        polygon_ids: lista de ids de los polígonos procesados
        """
        try:
            if not self.workflow_dict:
                logger.error("No hay workflow_dict inicializado para actualizar resultados OCR.")
                return False
                
            logger.info(f"DataFormatter recibe: {len(final_results)} resultados, {len(polygon_ids)} IDs")

            for idx, res in enumerate(final_results):
                if idx < len(polygon_ids) and res is not None:
                    poly_id = polygon_ids[idx]
                    if poly_id in self.workflow_dict["polygons"]:
                        self.workflow_dict["polygons"][poly_id]["ocr_text"] = res.get("text", "")
                        self.workflow_dict["polygons"][poly_id]["ocr_confidence"] = res.get("confidence")
                        
            logger.debug("Texto actualizado")
            return True
        except Exception as e:
            logger.error(f"Error actualizando resultados OCR: {e}")
            return False
        
    def create_text_lines(self, lines_info: Dict[str, Any]) -> bool:
        """
        Guarda las líneas reconstruidas usando operaciones vectorizadas en el workflow_dict bajo la clave 'all_lines'.
        lines_info debe tener la estructura esperada por el esquema.
        """
        try:
            if not self.workflow_dict:
                logger.error("No hay workflow_dict inicializado para guardar líneas de texto.")
                return False

            # Filtrar líneas válidas
            valid_lines = {k: v for k, v in lines_info.items() if v is not None}
            
            if not valid_lines:
                logger.warning("No hay líneas válidas para procesar.")
                return True

            # Procesar geometría de líneas con arrays NumPy
            all_lines: Dict[str, Any] = {}
            for line_id, line_data in valid_lines.items():
                # Procesar bboxes y centroides con NumPy si están disponibles
                line_bbox = line_data.get("line_bbox", [])
                line_centroid = line_data.get("line_centroid", [])
                
                if line_bbox and len(line_bbox) == 4:
                    # Convertir a array para validación y procesamiento
                    bbox_array = np.array(line_bbox, dtype=np.float32)
                    line_bbox = bbox_array.tolist()  # Convertir de vuelta a lista para schema
                
                if line_centroid and len(line_centroid) == 2:
                    # Convertir a array para validación y procesamiento
                    centroid_array = np.array(line_centroid, dtype=np.float32)
                    line_centroid = centroid_array.tolist()  # Convertir de vuelta a lista para schema
                    
                all_lines[line_id] = {
                    "lineal_id": line_id,
                    "text": line_data.get("text", ""),
                    "polygon_ids": line_data.get("polygon_ids", []),
                    "line_bbox": line_bbox,
                    "line_centroid": line_centroid
                }
                
            self.workflow_dict["all_lines"] = all_lines
            num_lines = len(all_lines)
            logger.debug(f"Guardadas {num_lines} líneas reconstruidas en el workflow_dict.")
            for line_id, line_data in all_lines.items():
                logger.debug(f"Línea {line_id}: {line_data.get('text', '')}")
            return True
        except Exception as e:
            logger.error(f"Error guardando líneas de texto: {e}", exc_info=True)
            return False
        
    def save_tabular_lines(self, table_detection_result: Dict[str, Any]) -> bool:
        """
        Guarda las líneas tabulares detectadas en el formato correcto, incluyendo el encabezado inmediato superior.
        """
        try:
            if not self.workflow_dict or "all_lines" not in self.workflow_dict:
                logger.error("No hay workflow_dict o all_lines para guardar líneas tabulares.")
                return False
            
            # Extraer las líneas detectadas del resultado
            table_line_ids = table_detection_result.get("table_lines", [])
            
            # Crear diccionario de líneas tabulares
            tabular_lines: Dict[str, Any] = {}
            header_line_id: Optional[str] = None
            
            # Determinar el encabezado si existe
            if table_line_ids:
                all_line_keys = list(self.workflow_dict["all_lines"].keys())
                first_table_idx = all_line_keys.index(table_line_ids[0])
                if first_table_idx > 0:
                    header_line_id = all_line_keys[first_table_idx - 1]
                    # Insertar encabezado al inicio si no está ya incluido
                    if header_line_id not in table_line_ids:
                        table_line_ids = [header_line_id] + table_line_ids
            
            for line_id in table_line_ids:
                if line_id in self.workflow_dict["all_lines"]:
                    line_data = self.workflow_dict["all_lines"][line_id]
                    tabular_lines[line_id] = {
                        "texto": line_data.get("text", ""),
                        "header_line": line_id == header_line_id if header_line_id else False
                    }
            
            # Guardar en workflow_dict
            self.workflow_dict["tabular_lines"] = tabular_lines
            num_tab_lines = len(tabular_lines)
            logger.debug(f"Guardadas {num_tab_lines} líneas tabulares en tabular_lines (incluyendo encabezado si corresponde)")
            for line_id, data in tabular_lines.items():
                logger.debug(f"Línea tabular: {line_id} - Texto: {data.get('texto', '')}")
            return True
            
        except Exception as e:
            logger.error(f"Error guardando líneas tabulares: {e}", exc_info=True)
            return False
            
    def save_structured_table(self, df: pd.DataFrame, columns: List[str], semantic_types: Optional[List[str]] = None) -> bool:
        try:
            self.structured_table = StructuredTable(df=df, columns=columns, semantic_types=semantic_types)
            return True
        except Exception as e:
            logger.error(f"Error guardando structured_table en memoria: {e}")
            return False