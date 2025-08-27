# core/domain/data_formatter.py
from core.domain.data_models import WORKFLOW_SCHEMA, WorkflowDict, DENSITY_ENCODER, StructuredTable, Geometry, Metadata, Polygons, CroppedGeometry, CroppedImage, AllLines, LineGeometry, TabularLines
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
            
            # Crear dataclasses anidadas con conversión a np.ndarray
            geometry = Geometry(
                polygon_coords=np.array(poly_data["geometry"]["polygon_coords"]),
                bounding_box=np.array(poly_data["geometry"]["bounding_box"]),
                centroid=np.array(poly_data["geometry"]["centroid"])
            )
            
            cropped_geo = CroppedGeometry(
                padd_centroid=np.array(poly_data["cropped_geometry"]["padd_centroid"]) if poly_data["cropped_geometry"]["padd_centroid"] else np.array([]),
                padding_coords=np.array(poly_data["cropped_geometry"]["padding_coords"]) if poly_data["cropped_geometry"]["padding_coords"] else np.array([]),
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
            
            polygons_dict: Dict[str, Dict[str, Polygons]] = {}
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
            
            # Actualizar 
            if self.workflow:
                self.workflow.polygons = polygons_dataclass
                
            logger.debug(f"Polígonos creados y validados: {len(polygons_dict)}")
            return True
            
        except Exception as e:
            logger.error(f"Error en create_polygon_dicts: {e}")
            return False

    def get_structured_table(self) -> Optional[pd.DataFrame]:
        return self.structured_table.df if self.structured_table else None

    def get_structured_semantic_types(self) -> Optional[List[str]]:
        return self.structured_table.semantic_types if self.structured_table else None
        
    def clear_cropped_images(self, polygon_ids: List[str]) -> bool:
        """Libera las imágenes recortadas de polígonos específicos para ahorrar memoria"""
        try:
            if not self.workflow:
                logger.error("No hay workflow inicializado para limpiar imágenes.")
                return False
                
            cleared_count = 0
            for poly_id in polygon_ids:
                if poly_id in self.workflow.polygons:
                    polygon = self.workflow.polygons[poly_id]
                    if polygon.cropped_img is not None:
                        # Crear nuevo polígono sin imagen para mantener immutabilidad
                        updated_polygon = Polygons(
                            polygon_id=polygon.polygon_id,
                            geometry=polygon.geometry,
                            cropedd_geometry=polygon.cropedd_geometry,
                            cropped_img=None,  # Limpiar imagen
                            perimeter=polygon.perimeter,
                            line_id=polygon.line_id,
                            ocr_text=polygon.ocr_text,
                            ocr_confidence=polygon.ocr_confidence,
                            was_fragmented=polygon.was_fragmented,
                            status=polygon.status,
                            stage=polygon.stage
                        )
                        self.workflow.polygons[poly_id] = updated_polygon
                        cleared_count += 1
                        
            # También limpiar del dict serializado
            for poly_id in polygon_ids:
                if poly_id in self.workflow_dict["polygons"]:
                    self.workflow_dict["polygons"][poly_id]["cropped_img"] = None
                    
            logger.debug(f"Liberadas {cleared_count} imágenes recortadas de memoria.")
            return True
        except Exception as e:
            logger.error(f"Error liberando imágenes recortadas: {e}")
            return False
            
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
        """Guarda imágenes recortadas y geometría de recorte en los polígonos del workflow_dict y las dataclasses"""
        try:
            if not self.workflow_dict or not self.workflow:
                logger.error("No hay workflow_dict o workflow inicializado para guardar imágenes recortadas.")
                return False

            for poly_id, img in cropped_images.items():
                if poly_id in self.workflow_dict["polygons"]:
                    self.workflow_dict["polygons"][poly_id]["cropped_img"] = img
                    if poly_id in cropped_geometries:
                        self.workflow_dict["polygons"][poly_id]["cropped_geometry"] = cropped_geometries[poly_id]

                # Actualizar también la dataclass
                if poly_id in self.workflow.polygons:
                    polygon = self.workflow.polygons[poly_id]
                    cropped_geo = cropped_geometries.get(poly_id)
                    
                    # Crear nuevo objeto CroppedImage
                    cropped_image_obj = CroppedImage(img)
                    
                    # Crear nuevo objeto CroppedGeometry
                    cropped_geometry_obj = CroppedGeometry(
                        padd_centroid=np.array(cropped_geo["padd_centroid"]) if cropped_geo and cropped_geo["padd_centroid"] else np.array([]),
                        padding_coords=np.array(cropped_geo["padding_coords"]) if cropped_geo and cropped_geo["padding_coords"] else np.array([]),
                        poly_dims=cropped_geo.get("poly_dims", {}) if cropped_geo else {}
                    )
                    
                    # Crear nuevo polígono con la imagen recortada y la geometría
                    updated_polygon = Polygons(
                        polygon_id=polygon.polygon_id,
                        geometry=polygon.geometry,
                        cropedd_geometry=cropped_geometry_obj,
                        cropped_img=cropped_image_obj,
                        perimeter=polygon.perimeter,
                        line_id=polygon.line_id,
                        ocr_text=polygon.ocr_text,
                        ocr_confidence=polygon.ocr_confidence,
                        was_fragmented=polygon.was_fragmented,
                        status=polygon.status,
                        stage=polygon.stage
                    )
                    self.workflow.polygons[poly_id] = updated_polygon

            logger.debug(f"Guardadas {len(cropped_images)} imágenes recortadas y geometría de recorte en workflow_dict y dataclasses.")
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
        # También actualizar la dataclass
        if self.workflow and poly_id in self.workflow.polygons:
            polygon = self.workflow.polygons[poly_id]
            updated_polygon = Polygons(
                polygon_id=polygon.polygon_id,
                geometry=polygon.geometry,
                cropedd_geometry=polygon.cropedd_geometry,
                cropped_img=CroppedImage(cropped_img) if cropped_img is not None else None,
                perimeter=polygon.perimeter,
                line_id=polygon.line_id,
                ocr_text=polygon.ocr_text,
                ocr_confidence=polygon.ocr_confidence,
                was_fragmented=polygon.was_fragmented,
                status=success,
                stage=worker_name
            )
            self.workflow.polygons[poly_id] = updated_polygon
            
    def update_ocr_results(self, final_results: List[Optional[Dict[str, Any]]], polygon_ids: List[str]) -> bool:
        """
        Actualiza los resultados de OCR en las dataclasses de polígonos.
        """
        try:
            if not self.workflow:
                logger.error("No hay workflow inicializado para actualizar resultados OCR.")
                return False
                
            logger.info(f"DataFormatter recibe: {len(final_results)} resultados, {len(polygon_ids)} IDs")

            for idx, res in enumerate(final_results):
                if idx < len(polygon_ids) and res is not None:
                    poly_id = polygon_ids[idx]
                    if poly_id in self.workflow.polygons:
                        # Actualizar la dataclass directamente
                        polygon = self.workflow.polygons[poly_id]
                        # Crear nuevo polígono con texto actualizado
                        updated_polygon = Polygons(
                            polygon_id=polygon.polygon_id,
                            geometry=polygon.geometry,
                            cropedd_geometry=polygon.cropedd_geometry,
                            cropped_img=polygon.cropped_img,
                            perimeter=polygon.perimeter,
                            line_id=polygon.line_id,
                            ocr_text=res.get("text", ""),  
                            ocr_confidence=res.get("confidence"), 
                            was_fragmented=polygon.was_fragmented,
                            status=polygon.status,
                            stage=polygon.stage
                        )
                        self.workflow.polygons[poly_id] = updated_polygon
                        
            logger.info("Texto OCR actualizado en dataclasses")
            return True
        except Exception as e:
            logger.error(f"Error actualizando resultados OCR: {e}")
            return False
        
    def create_text_lines(self, lines_info: Dict[str, Any]) -> bool:
        """
        Guarda las líneas reconstruidas en el workflow_dict y, más importante,
        crea las dataclasses AllLines y las guarda en el workflow (la fuente de verdad).
        """
        try:
            if not self.workflow:
                logger.error("No hay workflow_dict o workflow inicializado para guardar líneas de texto.")
                return False

            valid_lines = {k: v for k, v in lines_info.items() if v is not None}
            if not valid_lines:
                logger.warning("No hay líneas válidas para procesar.")
                return True

            # --- LÓGICA NUEVA PARA DATACLASSES ---
            all_lines_dataclasses: Dict[str, AllLines] = {}
            for line_id, line_data in valid_lines.items():
                line_geometry = LineGeometry(
                    line_centroid=line_data.get("line_centroid", [0, 0]),
                    line_bbox=line_data.get("line_bbox", [0, 0, 0, 0])
                )
                
                # La codificación de texto se hará después en get_encode_lines
                all_lines_dataclasses[line_id] = AllLines(
                    lineal_id=line_id,
                    text=line_data.get("text", ""),
                    encoded_text=[], 
                    polygon_ids=line_data.get("polygon_ids", []),
                    line_geometry=line_geometry,
                    tabular_line=False # Se determinará en etapas posteriores
                )
            
            # Actualiza la fuente de verdad (dataclasses)
            self.workflow.all_lines = all_lines_dataclasses
            
            # Mantenemos la actualización del diccionario por ahora para compatibilidad
            self.workflow_dict["all_lines"] = {
                line_id: {
                    "lineal_id": data.lineal_id,
                    "text": data.text,
                    "polygon_ids": data.polygon_ids,
                    "line_bbox": data.line_geometry.line_bbox,
                    "line_centroid": data.line_geometry.line_centroid
                } for line_id, data in all_lines_dataclasses.items()
            }

            num_lines = len(all_lines_dataclasses)
            logger.debug(f"Guardadas {num_lines} líneas reconstruidas en workflow_dict y dataclasses.")
            for line_id, line_data in self.workflow_dict["all_lines"].items():
                logger.debug(f"Línea {line_id}: {line_data.get('text', '')}")
            return True
        except Exception as e:
            logger.error(f"Error guardando líneas de texto: {e}", exc_info=True)
            return False        
            
        # Reemplaza la función save_tabular_lines (aprox. línea 501)

    def save_tabular_lines(self, table_detection_result: Dict[str, Any]) -> bool:
        """
        Identifica las líneas tabulares y las guarda como dataclasses TabularLines
        en el workflow. También actualiza el flag en AllLines.
        """
        try:
            if not self.workflow or not self.workflow.all_lines:
                logger.error("No hay workflow o all_lines para guardar líneas tabulares.")
                return False
            
            table_line_ids = table_detection_result.get("table_lines", [])
            if not table_line_ids:
                logger.info("No se detectaron líneas tabulares.")
                return True # No es un error, simplemente no hay tablas

            tabular_lines_dataclasses: Dict[str, TabularLines] = {}
            
            # 1. Poblar las dataclasses TabularLines y actualizar el flag en AllLines
            for line_id in table_line_ids:
                if line_id in self.workflow.all_lines:
                    line_obj = self.workflow.all_lines[line_id]
                    line_obj.tabular_line = True  # Actualiza la fuente de verdad
                    
                    # Crea la nueva dataclass para la línea tabular
                    tabular_lines_dataclasses[line_id] = TabularLines(
                        lineal_id=line_id,
                        complete_text=line_obj.text
                    )

            # 2. Asignar las nuevas dataclasses al workflow
            self.workflow.tabular_lines = tabular_lines_dataclasses
            
            # 3. Mantener el diccionario sincronizado para compatibilidad
            self.workflow_dict["tabular_lines"] = {
                line_id: {"text": data.complete_text} 
                for line_id, data in tabular_lines_dataclasses.items()
            }

            num_tab_lines = len(self.workflow.tabular_lines)
            logger.debug(f"Guardadas {num_tab_lines} líneas tabulares en dataclasses y dict.")
            for line_id, data in self.workflow.tabular_lines.items():
                logger.debug(f"Línea tabular: {line_id} - Texto: {data.complete_text}")
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