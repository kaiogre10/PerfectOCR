# core/domain/data_formatter.py
from core.domain.data_models import WORKFLOW_SCHEMA, WorkflowDict, DENSITY_ENCODER, CHAR_FRECUENCY, StructuredTable, Geometry, Metadata, Polygons, CroppedGeometry, CroppedImage, AllLines, LineGeometry
import numpy as np
import dataclasses
import jsonschema
import logging
import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd #type: ignore

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
        self.frecuency = CHAR_FRECUENCY
        self.structured_table: Optional[StructuredTable] = None

    def create_dict(self, IDRegistro: str, full_img: np.ndarray[Any, np.dtype[np.uint8]], metadata: Dict[str, Any]) -> bool:
        """Crea un nuevo dict"""
        self.temp_dict: Dict[str, Any] = {
            "IDRegistro": IDRegistro,
            "full_img": [full_img.tolist() if hasattr(full_img, 'tolist') else full_img],
            "global_data": {},
            "metadata": {
                "image_name": str(metadata.get("image_name", "")),
                "format": str(metadata.get("format", "")),
                "img_dims": {
                    "width": float(metadata.get("img_dims", {}).get("width")),
                    "height": float(metadata.get("img_dims", {}).get("height")),
                    "size": float(metadata.get("img_dims", {}).get("size")),
                },
                "date_creation": metadata.get("date_creation", datetime.now().isoformat()),
            },
            "polygons": {},
            "all_lines": {},
        }

        try:
            jsonschema.validate(self.temp_dict, self.schema)
            metadata_obj = Metadata(
                image_name=self.temp_dict["metadata"]["image_name"],
                format=self.temp_dict["metadata"]["format"],
                img_dims=self.temp_dict["metadata"]["img_dims"],
                date_creation=self.temp_dict["metadata"]["date_creation"],
            )

            self.workflow = WorkflowDict(
                IDRegistro=IDRegistro,
                full_img=full_img,
                global_data={},
                metadata=metadata_obj,
                polygons={},
                all_lines={},
            )
            self.workflow_dict = self.temp_dict

            return True
        except Exception as e:
            logger.error(f"Error validando workflow_dict: {e}", exc_info=True)
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
                croppy_dims=poly_data["cropped_geometry"].get("croppy_dims", {})
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
                stage=poly_data.get("stage", ""),
                key_field=poly_data.get("key_field", "global"),
                semantic_type=poly_data.get("semantic_type", "")
            )
            
            return polygon
        except jsonschema.ValidationError as e:
            logger.error(f"Polígono no válido: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Error creando polígono: {e}", exc_info=True)
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
                        "centroid": centroid.tolist(),
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
                    "stage": "",
                    "key_field": "",
                    "semantic_type": ""
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
                            stage=polygon.stage,
                            key_field=polygon.key_field,
                            semantic_type=polygon.semantic_type,
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
            logger.error(f"Error liberando imágenes recortadas: {e}", exc_info=True)
            return False
            
    def get_encode_lines(self, line_ids: Optional[List[str]] = None) -> Dict[str, List[int]]:
        """
        Codifica líneas específicas usando DENSITY_ENCODER con operaciones optimizadas.
        Si no se especifican line_ids, codifica todas las líneas existentes.
        """
        try:
            if not self.workflow or not hasattr(self.workflow, "all_lines") or not self.workflow.all_lines:
                logger.warning("No hay líneas disponibles para codificar.")
                return {}
                
            encoded_lines: Dict[str, List[int]] = {}
            all_lines: Dict[str, Any] = self.workflow.all_lines
            lines_to_encode = line_ids if line_ids is not None else list(all_lines.keys())
            
            for line_id in lines_to_encode:
                if line_id in all_lines:
                    line_obj = all_lines[line_id]
                    line_text = getattr(line_obj, "text", "")
                    if line_text:
                        compact_text = ''.join(line_text.split())
                        encoded_text = [self.encoder.get(char, 0) for char in compact_text]
                        encoded_lines[line_id] = encoded_text
                    else:
                        logger.warning(f"Línea {line_id} no tiene texto para codificar.")
                else:
                    logger.warning(f"Línea {line_id} no encontrada en all_lines.")
            
            logger.debug(f"Codificadas {len(encoded_lines)} líneas para análisis de densidad.")
            return encoded_lines
        except Exception as e:
            logger.error(f"Error codificando líneas: {e}", exc_info=True)
            return {}
            
    def get_tabular_lines(self) -> List[str]:
        """
        Retorna la lista de line_id marcadas como tabulares en workflow.all_lines.
        Devuelve lista vacía si no hay workflow o no hay líneas marcadas.
        """
        try:
            if not self.workflow or not getattr(self.workflow, "all_lines", None):
                logger.debug("get_tabular_lines: No hay workflow o all_lines vacío.")
                return []

            tabular_ids: List[str] = []
            for line_id, line_obj in self.workflow.all_lines.items():
                try:
                    if getattr(line_obj, "tabular_line", False):
                        tabular_ids.append(line_id)
                except Exception:
                    continue

            logger.debug(f"get_tabular_lines: encontradas {len(tabular_ids)} líneas tabulares.")
            return tabular_ids
        except Exception as e:
            logger.error(f"Error obteniendo lineas tabulares: {e}", exc_info=True)
            return []
                
    def get_frecuency_char(self) -> Optional[Dict[str, int]]:
        """Obtiene los valores de frecuencia para letras"""
        try:
            if self.frecuency:
                return self.frecuency
        except Exception as e:
            logger.warning(f"Error entregando frecuencias: {e}", exc_info=True)

    def update_full_img(self, full_img: (Optional[np.ndarray[Any, np.dtype[np.uint8]]])=None) -> bool:
        """Actualiza o vacía la imagen completa en el workflow"""
        try:
            if not self.workflow:
                logger.error("No hay workflow inicializado para actualizar full_img.")
                return False
                
            if full_img is None:
                # Si se pasa None, vaciamos la imagen para liberar memoria
                dataclasses.replace(self.workflow, full_img=None)
                logger.debug(f"Imagen liberada con éxito: {full_img}")
            return True
        except Exception as e:
            logger.error(f"Error actualizando full_img: {e}", exc_info=True)
            return False
            
    def save_cropped_images(
    self,
    cropped_images: Dict[str, np.ndarray[Any, np.dtype[np.uint8]]],
    cropped_geometries: Dict[str, Dict[str, Any]]
    ) -> bool:
        """Guarda imágenes recortadas y geometría de recorte en los polígonos de las dataclasses"""
        try:
            if not self.workflow:
                logger.error("No hay workflow inicializado para guardar imágenes recortadas.")
                return False

            for poly_id, img in cropped_images.items():
                if poly_id in self.workflow.polygons:
                    polygon = self.workflow.polygons[poly_id]
                    cropped_geo = cropped_geometries.get(poly_id)
                    
                    # Crear nuevo objeto CroppedImage
                    cropped_image_obj = CroppedImage(img)
                    
                    # Crear nuevo objeto CroppedGeometry
                    cropped_geometry_obj = CroppedGeometry(
                        padd_centroid=np.array(cropped_geo["padd_centroid"]) if cropped_geo and cropped_geo["padd_centroid"] else np.array([]),
                        padding_coords=np.array(cropped_geo["padding_coords"]) if cropped_geo and cropped_geo["padding_coords"] else np.array([]),
                        croppy_dims=cropped_geo.get("croppy_dims", {}) if cropped_geo else {}
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
                        stage=polygon.stage,
                        key_field=polygon.key_field,
                        semantic_type=polygon.semantic_type,
                    )
                    self.workflow.polygons[poly_id] = updated_polygon

            logger.debug(f"Guardadas {len(cropped_images)} imágenes recortadas y geometría de recorte en dataclasses.")
            return True
        except Exception as e:
            logger.error(f"Error guardando imágenes recortadas y geometría: {e}", exc_info=True)
            return False
        
    def update_preprocessing_result(self, poly_id: str, cropped_img: np.ndarray[Any, np.dtype[np.uint8]], worker_name: str) -> bool:
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
                status=polygon.status,
                stage=polygon.stage,
                key_field=polygon.key_field,
                semantic_type=polygon.semantic_type,
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
                
            logger.debug(f"DataFormatter recibe: {len(final_results)} resultados, {len(polygon_ids)} IDs")
 
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
                            stage=polygon.stage,
                            key_field=polygon.key_field,
                            semantic_type=polygon.semantic_type,
                        )
                        
                        self.workflow.polygons[poly_id] = updated_polygon
                        
            logger.debug("Texto OCR actualizado en dataclasses")
            return True
        except Exception as e:
            logger.error(f"Error actualizando resultados OCR: {e}", exc_info=True)
            return False

    def update_semantic_type(self, final_results: Dict[str, str]) -> bool:
        try:
            if not self.workflow:
                logger.error("No hay workflow inicializado para actualizar resultados OCR.")
                return False

            updated_count = 0

            for poly_id, semantic_type in final_results.items():
                if poly_id in self.workflow.polygons:
                    polygon = self.workflow.polygons[poly_id]

                    updated_polygon = dataclasses.replace(polygon, semantic_type=semantic_type)
                    self.workflow.polygons[poly_id] = updated_polygon
                    updated_count += 1
            
                    logger.debug(f"{poly_id}, semantic={semantic_type}, text='{polygon.ocr_text or ''}'")

            if updated_count > 0:
                logger.debug(f"Actualizados {updated_count} polígonos con semantic_types")
            return True
            
        except Exception as e:
            logger.error(f"Error actualizando múltiples polígonos: {e}", exc_info=True)
            return False
        
    def update_polygon_data(self, polygon_updates: Optional[Dict[str, str]]) -> bool:
        """
        Actualiza los datos de los polígonos en las dataclasses de polígonos.
        """
        try:
            if not self.workflow:
                logger.error("No hay workflow inicializado para actualizar polígonos.")
                return False
            if not polygon_updates:
                return False

            updated_count = 0

            for poly_id, key_field in polygon_updates.items():
                if poly_id in self.workflow.polygons:
                    polygon = self.workflow.polygons[poly_id]

                    updated_polygon = dataclasses.replace(polygon, key_field=key_field)
                    self.workflow.polygons[poly_id] = updated_polygon
                    updated_count += 1
            
                    logger.debug(f"UPDATED: poly_id={poly_id}, key_field={key_field}, text='{polygon.ocr_text or ''}'")

            if updated_count > 0:
                logger.debug(f"Actualizados {updated_count} polígonos con key_fields")
            return True
            
        except Exception as e:
            logger.error(f"Error actualizando múltiples polígonos: {e}", exc_info=True)
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
            
            if not lines_info:
                return False

            valid_lines = {k: v for k, v in lines_info.items() if v is not None}
            if not valid_lines:
                logger.warning("No hay líneas válidas para procesar.")
                return False
                
            all_lines_dataclasses: Dict[str, AllLines] = {}
            tabular_lines_info: List[Dict[str, Any]] = []
            for line_id, line_data in valid_lines.items():
                line_geometry = LineGeometry(
                    line_centroid=line_data.get("line_centroid", [0, 0]),
                    line_bbox=line_data.get("line_bbox", [0, 0, 0, 0])
                )
                
                all_lines_dataclasses[line_id] = AllLines(
                    lineal_id=line_id,
                    text=line_data.get("text", ""),
                    encoded_text=[], 
                    polygon_ids=line_data.get("polygon_ids", []),
                    line_geometry=line_geometry,
                    tabular_line=False,
                    header_line=False,
                )
            
            self.workflow.all_lines = all_lines_dataclasses

            for line_id in self.workflow.all_lines:
                if line_id in self.workflow.all_lines:
                    line_obj = self.workflow.all_lines[line_id]
                    tabular_lines_info.append({
                                "line_id": line_id,
                                "text": line_obj.text,
                            })

            if tabular_lines_info:
                for all_lines in tabular_lines_info:
                    logger.info(f"Linea textual: {all_lines['line_id']}: '{all_lines['text']}'")
            
            num_lines = len(all_lines_dataclasses)
            logger.debug(f"Guardadas {num_lines} líneas reconstruidas en dataclasses.")
            for line_id, line_data in self.workflow.all_lines.items():
                return True
        except Exception as e:
            logger.error(f"Error guardando líneas de texto: {e}", exc_info=True)
            return False
        
    def update_header(self, header_line_id: str) -> bool:
        try:
            if not self.workflow:
                logger.error("No hay workflow inicializado para actualizar encabezado.")
                return False
            
            # verificar que la línea existe
            if header_line_id not in self.workflow.all_lines:
                logger.error(f"Línea {header_line_id} no encontrada en all_lines.")
                return False
            
            # actualizar el flag header_line a True para la línea especificada
            current_line = self.workflow.all_lines[header_line_id]
            updated_line = dataclasses.replace(current_line, header_line=True)
            self.workflow.all_lines[header_line_id] = updated_line
            
            logger.info(f"Línea {header_line_id} marcada como header.")
            return True
            
        except Exception as e:
            logger.error(f"Error actualizando header: {e}", exc_info=True)
            return False

    def save_tabular_lines(self, line_ids: List[str]) -> bool:
        """
        Identifica las líneas tabulares y las guarda como dataclasses TabularLines
        en el workflow. También actualiza el flag en AllLines.

        Comportamiento: cada llamada primero borra todos los flags tabular_line
        y después marca únicamente los line_ids provistos.
        """
        try:
            if not self.workflow:
                return False
    
            # 1) Limpiar todos los flags tabular_line existentes
            cleared_count = 0
            for lid, line_obj in self.workflow.all_lines.items():
                try:
                    if getattr(line_obj, "tabular_line", False):
                        updated = dataclasses.replace(line_obj, tabular_line=False)
                        self.workflow.all_lines[lid] = updated
                        cleared_count += 1
                except Exception:
                    # Fallback: intentar asignar directamente si dataclasses.replace falla
                    try:
                        line_obj.tabular_line = False
                        cleared_count += 1
                    except Exception:
                        continue

            logger.debug(f"Limpiados {cleared_count} flags tabular_line previos.")

            # 2) Marcar los nuevos line_ids provistos (si los hay)
            if not line_ids:
                logger.warning("No se recibieron line_ids en el manager.")
            marked_count = 0
            tabular_lines_info: List[Dict[str, Any]] = []
            marked_ids: List[str] = []
            for line_id in line_ids:
                if line_id in self.workflow.all_lines:
                    line_obj = self.workflow.all_lines[line_id]

                    updated_line = dataclasses.replace(line_obj, tabular_line=True)
                    self.workflow.all_lines[line_id] = updated_line
                    marked_count += 1
                    marked_ids.append(line_id)
                    tabular_lines_info.append({
                        "line_id": line_id,
                        "text": getattr(self.workflow.all_lines[line_id], "text", ""),
                        "polygon_ids": getattr(self.workflow.all_lines[line_id], "polygon_ids", [])
                    })

            if marked_ids:
                logger.debug(f"Marcadas {marked_count} líneas como tabulares: {marked_ids}")
                for log_info in tabular_lines_info:
                    logger.info(f"Líneas tabulares: {log_info['line_id']}: '{log_info['text']}' | polygons: {log_info['polygon_ids']}")
            else:
                logger.warning("No se marcaron líneas como tabulares en esta llamada a save_tabular_lines.")

            # Operación completada (se limpió y se marcaron las nuevas líneas)
            return True
        except Exception as e:
            logger.error(f"Error marcando líneas como tabulares: {e}", exc_info=True)
            return False
                
    def save_structured_table(self, df: pd.DataFrame, columns: List[str], semantic_types: Optional[List[str]] = None) -> bool:
        try:
            self.structured_table = StructuredTable(df=df, columns=columns, semantic_types=semantic_types)
            return True
        except Exception as e:
            logger.error(f"Error guardando structured_table en memoria: {e}", exc_info=True)
            return False

    def _parse_number(self, s: Any):
        try:
            if s is None:
                return None
            if isinstance(s, (int, float)):
                return s
            sstr = str(s).replace(",", "").strip()
            if sstr == "":
                return None
            if "." in sstr:
                return float(sstr)
            return int(sstr)
        except Exception:
            try:
                return float(str(s).replace(",", ""))
            except Exception:
                return None

    def _row_to_detalle(self, row: Dict[str, Any]) -> Dict[str, Any]:
        # Mapea columnas esperadas al esquema de DetallesCompra
        return {
            "IDDetalle": None,  
            "IDRegistro": None,
            "Cantidad": self._parse_number(row.get("c")),
            "SKU": row.get("sku"),
            "ProductoEstandarizado": None,
            "PrecioUnitario": self._parse_number(row.get("pu")),
            "ImporteRaw": self._parse_number(row.get("mtl")),
        }
    
    def to_db_payload(self, data_base_path: str) -> str:
        """
        Construye un payload JSON-serializable:
        { registro: {...}, detalles: [...], provenance: {...}, raw_table: [...] }
        """
        try:
            t0 = time.perf_counter()
            logger.debug("Estructuración de datos iniciada")
            try:
                if not self.workflow:
                    return {}

                wf: WorkflowDict = self.workflow
                md: Metadata = wf.metadata
                polygons: Dict[str, Polygons] = wf.polygons or {}
                dict_id: str = wf.IDRegistro
                folio: Optional[str] = None
                fecha: Optional[str] = None
                rfc: Optional[str] = None
                monto: Optional[Any] = None
                tipo: Optional[str] = None
                nombre_cliente: Optional[str] = None

                for pid, pdata in polygons.items():
                    try:
                        key = getattr(pdata, "key_field", None)
                        text = getattr(pdata, "ocr_text", None)
                        if not key or not text:
                            continue
                        key = str(key).strip()
                        txt = str(text).strip()
                        if key == "FolioDocumento" and not folio:
                            folio = txt
                        elif key == "FechaDocumento" and not fecha:
                            try:
                                parsed = datetime.fromisoformat(txt)
                                fecha = parsed.isoformat()
                            except Exception:
                                fecha = txt
                        elif key == "RFCProveedor" and not rfc:
                            rfc = txt
                        elif key == "MontoTotalDocumento" and monto is None:
                            monto = self._parse_number(txt)
                        elif key == "TipoDocumento" and not tipo:
                            tipo = txt
                        elif key == "NombreCliente" and not nombre_cliente:
                            nombre_cliente = txt
                    except Exception as e:
                        # No bloquear la construcción del payload por un polígonos con problemas
                        logger.debug(f"Ignorando polígono {pid} al construir registro {e}", exc_info=True)

                registro: Optional[Dict[str, Any]] = {
                    "IDRegistro": dict_id,
                    "FolioDocumento": folio,
                    "FechaDocumento": fecha,
                    "RFCProveedor": rfc,
                    "MontoTotalDocumento": monto,
                    "TipoDocumento": tipo,
                    "NombreCliente": nombre_cliente,
                }

            except Exception as e:
                logger.debug(f"fallo en dbpayload{e}", exc_info=True)
            try:
                detalles: List[List[int]] = []
                if self.structured_table and hasattr(self.structured_table, "df"):
                    df = self.structured_table.df
                    cols = self.structured_table.columns if self.structured_table.columns else list(df.columns)
                    for _, r in df.iterrows():
                        row: Dict[str, Any] = {}
                        # mapear por índice de columnas a col_0..col_n para _row_to_detalle
                        for i, c in enumerate(cols):
                            row[f"col_{i}"] = r.get(c) if c in df.columns else r[i]
                        detalles.append(self._row_to_detalle(row))

            except Exception as e:
                logger.debug(f"No hay tabla estructurada{e}", exc_info=True)
                            
            provenance: Dict[str, Any] = {
                "IDRegistro": dict_id,
                "IDProveedor": md.get("image_name") if isinstance(md, dict) else getattr(md, "image_name", None)
            }

            payload: Dict[str, Any] = {"registro": registro, "detalles": detalles, "provenance": provenance}
            # opcional: raw_table para auditoría
            if self.structured_table and hasattr(self.structured_table, "df"):
                payload["raw_table"] = {"columns": list(self.structured_table.df.columns), "rows": self.structured_table.df.fillna("").astype(str).values.tolist()}

            success: bool = self.export_payload_json(payload, data_base_path)
            if success is not False:
                logger.debug(f"Estructuración de datos completada en {time.perf_counter()-t0:.6f}s")
                logger.debug(f"Resultados: {payload}")
                db_path = data_base_path
                return db_path

        except Exception as e:
            logger.error(f"Error exportando payload json: {e}", exc_info=True)

    def export_payload_json(self, payload: Dict[str, Any], data_base_path: str) -> bool:
        """Escribe el payload en disco para auditoría/revisión manual"""
        try:
            if not payload:
                return False
            with open(data_base_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
                
            return True
        except Exception as e:
            logger.debug(f"Error exportando payload json: {e}", exc_info=True)
            return False

