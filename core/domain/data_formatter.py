# core/domain/data_formatter.py
from core.utils.data_utils import DENSITY_ENCODER, CHAR_FRECUENCY, INV_FRECUENCY_ENCODER, SEMATIC_TYPES_MAP
from core.domain.data_models import WorkflowDict, StructuredTable, Geometry, Metadata, Polygons, CroppedGeometry, CroppedImage, AllLines, LineGeometry, FullImage
import numpy as np
import dataclasses
import logging
import json
import time
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from core.utils.image_utils import normalice_image
import pandas as pd #type: ignore

logger = logging.getLogger(__name__)

class DataFormatter:
    """Válvula de entrada/salida para todas las operaciones del workflow."""
    def __init__(self):
        self.workflow: Optional[WorkflowDict] = None
        self.density_encoder: Optional[Dict[str, float]] = None
        self.frecuency_encoder: Optional[Dict[str, float]] = None
        self.inv_frecuency: Optional[Dict[str, float]] = None
        self.mean_dummie: Optional[Dict[str, float]] = None
        self.median_dummie: Optional[Dict[str, float]] = None
        self.structured_table: Optional[StructuredTable] = None
        self.semantic_map: Optional[Dict[str, int]] = None
    
    def create_workflow(self, IDRegistro: str, gray_img: np.ndarray[Any, np.dtype[np.uint8]], metadata: Dict[str, Any]) -> bool:
        """Crea un nuevo workflow usando dataclasses"""
        try:
            full_image = FullImage(
                full_img=(gray_img)
                )
            
            metadata_obj = Metadata(
                image_name=str(metadata.get("image_name", "")),
                date_creation=str(metadata.get("date_creation" or "")),
                dpi=int(metadata.get("dpi", {})),
                img_dims=tuple(metadata["img_dims"])
            )

            self.workflow = WorkflowDict(
                IDRegistro=IDRegistro,
                full_img=full_image,
                metadata=metadata_obj,
                polygons={},
                all_lines={}
            )
            return True
            
        except Exception as e:
            logger.error(f"No se pudo crear el workflowDict: {e}", exc_info=True)
        return False
    
    def create_polygon_dicts(self, results: Dict[str, Dict[str, Any]]) -> bool:
        """Refactorizado para usar validación + dataclasses"""
        try:
            if self.workflow is None:
                return False
            
            polygons_dataclass: Dict[str, Polygons] = {}
            
            for pid, poly_data in results.items():
                poly_id = pid
                poly_index = poly_data.get("poly_index", 0)
                coords = poly_data["polygon_coords"]
                bbox = poly_data["bounding_box"]
                centroid = poly_data["centroid"]

                # Crear objeto Geometry
                geometry = Geometry(
                    polygon_coords=coords,
                    bounding_box=bbox,
                    centroid=centroid
                )

                # Crear objeto Polygons y agregar al diccionario
                polygon_obj = Polygons(
                    polygon_id=poly_id,
                    poly_index=poly_index,
                    geometry=geometry,
                    cropedd_geometry=None, #type:ignore
                    cropped_img=None,
                    ocr_text=None,
                    ocr_confidence=None,
                    was_fragmented=False,
                    key_field=None,
                    semantic_clasification=0,
                )
                polygons_dataclass[poly_id] = polygon_obj
                                
            # Actualizar 
            if self.workflow:
                self.workflow.polygons = polygons_dataclass
                
            logger.debug(f"Polígonos creados y validados: {len(polygons_dataclass)}")
            return True
            
        except Exception as e:
            logger.error(f"Error en create_polygon_dicts: {e}", exc_info=True)
        return False
            
    def get_full_img(self) -> Optional[FullImage]:
        return self.workflow.full_img if self.workflow else None
            
    def get_structured_table(self) -> Optional[pd.DataFrame]:
        return self.structured_table.df if self.structured_table else None
        
    def delete_cropped_images(self) -> bool:
        """Libera todas las imágenes recortadas de los polígonos para ahorrar memoria."""
        try:
            if not self.workflow or not self.workflow.polygons:
                logger.error("No hay workflow inicializado para limpiar imágenes recortadas.")
                return False

            for poly_id, polygon in self.workflow.polygons.items():
                updated_polygon = dataclasses.replace(polygon, cropped_img=None)
                self.workflow.polygons[poly_id] = updated_polygon

            logger.debug("Todas las imágenes recortadas han sido liberadas de memoria.")
            return True
        except Exception as e:
            logger.error(f"Error liberando imágenes recortadas: {e}", exc_info=True)
            return False

    def get_density_encoder(self) -> Dict[str, float]:
        """
        Obtiene los valores de densidad por letra.
        """
        try:
            if self.density_encoder is None:
                self.density_encoder = DENSITY_ENCODER
            return self.density_encoder
        
        except Exception as e:
            logger.warning(f"Error entregando frecuencias: {e}", exc_info=True)
            return {}

    def get_frecuency_char(self) -> Dict[str, float]:
        """
        Obtiene los valores de frecuencia por letra.
        """
        try:
            if self.frecuency_encoder is None:
                self.frecuency_encoder = CHAR_FRECUENCY
            return self.frecuency_encoder
        
        except Exception as e:
            logger.warning(f"Error entregando frecuencias: {e}", exc_info=True)
            return {}

    def get_inverse_frecuency_encoder(self) -> Dict[str, float]:
        """
        Obtiene los valores densos de frecuencia inversa por letra.
        """
        try:
            if self.inv_frecuency is None:
                self.inv_frecuency = INV_FRECUENCY_ENCODER
            return self.inv_frecuency
        
        except Exception as e:
            logger.warning(f"Error entregando frecuencias: {e}", exc_info=True)
            return {}

    def get_semmantic_types(self) -> Dict[str, int]:
        """
        Obtiene los valores morfológicos por letra.
        """
        try:
            if self.semantic_map is None:
                self.semantic_map = SEMATIC_TYPES_MAP
            return self.semantic_map
        
        except Exception as e:
            logger.warning(f"Error entregando frecuencias: {e}", exc_info=True)
            return {}

    def get_tabular_lines(self, return_objects: bool) -> Union[Dict[str, Any], List[str]] | List[str]:
        """
        Retorna las líneas marcadas como tabulares en workflow.all_lines.
        Args:
            return_objects: Si True, devuelve Dict[str, Any] con objetos completos.
            Si False, devuelve List[str] con solo los line_ids.
        Returns:
            Dict[str, Any] o List[str] según el parámetro return_objects.
            Devuelve estructura vacía si no hay workflow o no hay líneas marcadas.
        """
        try:
            if not self.workflow or not getattr(self.workflow, "all_lines", None):
                logger.debug("get_tabular_lines: No hay workflow o all_lines vacío.")
                return {} if return_objects else []

            tabular_lines: Dict[str, Any] = {}
            tabular_ids: List[str] = []
            
            for line_id, line_obj in self.workflow.all_lines.items():
                try:
                    if getattr(line_obj, "tabular_line", False):
                        tabular_lines[line_id] = line_obj
                        tabular_ids.append(line_id)
                except Exception:
                    continue

            logger.debug(f"Encontradas: {len(tabular_lines)} líneas tabulares.")
            
            if return_objects:
                return tabular_lines
            else:
                return tabular_ids
                
        except Exception as e:
            logger.error(f"Error obteniendo lineas tabulares: {e}", exc_info=True)
            return {} if return_objects else []

    def update_full_img(self, corrected: bool, full_img: Optional[np.ndarray[Any, np.dtype[np.uint8]]]=None) -> bool:
        """Actualiza o vacía la imagen completa en el workflow"""
        try:
            if not self.workflow:
                logger.error("No hay workflow inicializado para actualizar full_img.")
                return False
                
            if full_img is None:
                # Si se pasa None, vaciamos la imagen para liberar memoria
                self.workflow = dataclasses.replace(self.workflow, full_img=None)
                return True
            if corrected:
            #     # Normalizar si se recibe la dataclass FullImage corregida
                if isinstance(full_img, FullImage):
                    img_arr = getattr(full_img, "full_img", None)
                else:
                    img_arr = full_img
                                    
                img_arr = normalice_image(full_img)
        
                # Wrap en la dataclass FullImage y actualizar workflow
                full_image_obj = FullImage(img_arr)
                self.workflow = dataclasses.replace(self.workflow, full_img=full_image_obj)
                logger.debug("Imagen actualizada con éxito.")
                return True
            
            else:
                logger.debug("Imagen completa sin modificaciones")
                return True
            
        except Exception as e:
            logger.error(f"Error actualizando full_img: {e}", exc_info=True)
            return False
                        
    def save_cropped_images(self, cropped_images: Dict[str, np.ndarray[Any, np.dtype[np.uint8]]], cropped_geometries: Dict[str, Dict[str, Any]]) -> bool:
        """Guarda imágenes recortadas y geometría de recorte en los polígonos de las dataclasses"""
        try:
            if not self.workflow:
                logger.error("No hay workflow inicializado para guardar imágenes recortadas.")
                return False

            total_img = len(cropped_images)
            total_geo = len(cropped_geometries)

            if total_img != total_geo:
                logger.error(f"El número de imágenes recortadas '{total_img}' no coincide con el número de geometrías recortadas: '{total_geo}'")
                return False

            logger.debug(f"'{total_img}' imágenes recortadas recibidas para guardar.")

            for poly_id, img in cropped_images.items():
                if poly_id in self.workflow.polygons:
                    polygon = self.workflow.polygons[poly_id]
                    cropped_geo = cropped_geometries.get(poly_id)

                    # Crear nuevo objeto CroppedImage
                    cropped_image_obj = CroppedImage(normalice_image(img))

                    # Crear nuevo objeto CroppedGeometry
                    cropped_geometry_obj = CroppedGeometry(
                        padd_centroid=np.array(cropped_geo["padd_centroid"]) if cropped_geo and cropped_geo["padd_centroid"] else np.array([]),
                        padding_coords=np.array(cropped_geo["padding_coords"]) if cropped_geo and cropped_geo["padding_coords"] else np.array([]),
                        croppy_dims=cropped_geo.get("croppy_dims", {}) if cropped_geo else {}
                    )

                    # Crear nuevo polígono con la imagen recortada y la geometría
                    updated_polygon = dataclasses.replace(
                        polygon,
                        cropped_img=cropped_image_obj,
                        cropedd_geometry=cropped_geometry_obj
                    )
                    self.workflow.polygons[poly_id] = updated_polygon

            logger.debug(f"Guardadas {len(cropped_images)} imágenes recortadas y geometría de recorte")
            return True
        except Exception as e:
            logger.error(f"Error guardando imágenes recortadas y geometría: {e}", exc_info=True)
            return False
            
    def validate_cropped_img(self) -> bool:
        """
        Valida automáticamente todas las imágenes recortadas y elimina las blancas/inválidas.
        Retorna True si hay workflow válido, False si no hay workflow.
        """
        if not self.workflow:
            logger.error("No hay workflow inicializado para validar imágenes.")
            return False
        
        white_poly_ids: List[str] = []
        
        # Detectar polígonos blancos/inválidos usando el normalicer
        for poly_id, polygon in self.workflow.polygons.items():
            cropped_img = polygon.cropped_img.cropped_img if polygon.cropped_img else None
            if normalice_image(cropped_img) is None:  # None = imagen inválida
                white_poly_ids.append(poly_id)
        
        if white_poly_ids:
            logger.info(f"Eliminando {len(white_poly_ids)} polígonos blancos/inválidos")
            
            for poly_id in white_poly_ids:
                if poly_id in self.workflow.polygons:
                    del self.workflow.polygons[poly_id]
            
            # Reindexar polígonos restantes (patrón de poly_gone)
            remaining_polygons = list(self.workflow.polygons.items())
            new_polygons: Dict[str, Polygons] = {}
            
            for idx, (old_id, poly_obj) in enumerate(remaining_polygons): # type: ignore
                new_id = f"poly_{idx:04d}"
                updated_poly_obj = dataclasses.replace(poly_obj, polygon_id=new_id)
                new_polygons[new_id] = updated_poly_obj
            
            self.workflow.polygons = new_polygons
            logger.debug(f"Reindexados {len(new_polygons)} polígonos válidos")
        
        return True
        
    def update_ocr_results(self, final_results: Dict[str, Dict[str, Any]]) -> bool:
        """
        Actualiza los resultados de OCR en las dataclasses de polígonos.
        """
        try:
            if not self.workflow:
                logger.error("No hay workflow inicializado para actualizar resultados OCR.")
                return False
            
            if not final_results:
                logger.error(f"No hay Texto OCR")
                return False
                
            logger.debug(f"Recibe: {len(final_results)} resultados IDs")

            new_index = 0
            for poly_id, res in final_results.items():
                if poly_id in self.workflow.polygons:
                    polygon = self.workflow.polygons[poly_id]
                    new_index += 1
                    updated_polygon = dataclasses.replace(
                        polygon,
                        poly_index = new_index,
                        ocr_text=res.get("text", ""),
                        ocr_confidence=res.get("confidence")
                    )

                    self.workflow.polygons[poly_id] = updated_polygon
                else:
                    logger.warning(f"Polígono {poly_id} no encontrado en workflow.polygons")

            logger.debug("Texto OCR actualizado")
            return True
        except Exception as e:
            logger.error(f"Error actualizando resultados OCR: {e}", exc_info=True)
            return False
                        
    def update_semantic_clasification(self, final_results: Dict[str, List[int] | int]) -> bool:
        """
        Actualiza el semantic_clasification de los polígonos.
        """
        try:
            if not self.workflow:
                logger.error("No hay workflow inicializado para actualizar resultados OCR.")
                return False

            updated_count = 0

            for poly_id, semantic_type in final_results.items():
                if poly_id in self.workflow.polygons:
                    polygon = self.workflow.polygons[poly_id]

                    # Actualizar semantic_clasification y opcionalmente resetear was_refined
                    updated_polygon = dataclasses.replace(
                        polygon, 
                        semantic_clasification=semantic_type,
                    )                    
                    self.workflow.polygons[poly_id] = updated_polygon
                    updated_count += 1

            if updated_count > 0:
                logger.debug(f"Actualizados {updated_count} polígonos con semantic_clasifications")
            return True
            
        except Exception as e:
            logger.error(f"Error actualizando múltiples polígonos: {e}", exc_info=True)
            return False
            
    def merge_semantics(self) -> bool:
        """
        Unifica los tipos semánticos en las dataclasses, convirtiendo
        todos los 'quantitative' (2) a 'numeric' (1).
        """
        if not self.workflow or not self.workflow.polygons:
            return False

        updated_count = 0
        for poly_id, polygon in self.workflow.polygons.items():
            if polygon.semantic_clasification == 2:  # quantitative
                updated_polygon = dataclasses.replace(polygon, semantic_clasification=1)  # numeric
                self.workflow.polygons[poly_id] = updated_polygon
                updated_count += 1

            if polygon.semantic_clasification == -2:  # umd
                updated_polygon = dataclasses.replace(polygon, semantic_clasification=-1)  # code
                self.workflow.polygons[poly_id] = updated_polygon
                updated_count += 1
        
        if updated_count > 0:
            logger.debug(f"Unificados {updated_count} polígonos de 'quantitative' a 'numeric'.")
        return True
                
    def update_key_field(self, polygon_updates: Optional[Dict[str, List[int] | int]]) -> bool:
        """
        Actualiza los datos de los polígonos en las dataclasses de polígonos.
        """
        try:
            if not self.workflow:
                logger.error("No hay workflow inicializado para actualizar polígonos.")
                return False
            
            if not polygon_updates:
                logger.warning("Sin Key_fields")
                return True

            updated_count = 0

            for poly_id, key_field in polygon_updates.items():
                if poly_id in self.workflow.polygons:
                    polygon = self.workflow.polygons[poly_id]

                    updated_polygon = dataclasses.replace(polygon, key_field=key_field)
                    self.workflow.polygons[poly_id] = updated_polygon
                    updated_count += 1
            
                    logger.info(f"UPDATED: poly_id: {poly_id}, key_field= '{key_field}', text='{polygon.ocr_text}'")

            if updated_count > 0:
                logger.debug(f"Actualizados {updated_count} polígonos con key_fields")
                return True
            
            else:
                logger.warning("No hubo poligonos con key_field")
                return False
            
        except Exception as e:
            logger.warning(f"Error actualizando múltiples polígonos: {e}", exc_info=True)
            return False

    def _update_line_attr(self, line_id: str, attr_name: str, value: Any) -> bool:
        """Actualiza un atributo de una línea en all_lines de forma segura."""
        if not self.workflow or line_id not in self.workflow.all_lines:
            return False
            
        current = self.workflow.all_lines[line_id]
        updated = dataclasses.replace(current, **{attr_name: value})
        self.workflow.all_lines[line_id] = updated
        return True

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
                logger.error("Sin líneas tabulares")
                return False

            valid_lines = {k: v for k, v in lines_info.items() if v is not None}
            if not valid_lines:
                logger.error("No hay líneas válidas para procesar.")
                return False
                
            all_lines_dataclasses: Dict[str, AllLines] = {}
            for line_id, line_data in valid_lines.items():
                line_geometry = LineGeometry(
                    line_centroid=line_data["line_centroid"] or [0, 0],
                    line_bbox=line_data["line_bbox"] or [0, 0, 0, 0]
                )
                
                all_lines_dataclasses[line_id] = AllLines(
                    lineal_id=line_id,
                    line_index=line_data.get("line_index", 0),
                    text=line_data.get("text", ""),
                    polygon_ids=line_data["polygon_ids"],
                    polygons_index=line_data["polygons_index"],
                    line_geometry=line_geometry,
                    tabular_line=line_data["tabular_line"],
                    header_line=line_data["header_line"],
                    footer_line=line_data["footer_line"]
                )
            
            self.workflow.all_lines = all_lines_dataclasses
            return True
                        
        except Exception as e:
            logger.error(f"Error guardando líneas de texto: {e}", exc_info=True)
            return False

    def save_tabular_lines(self, line_ids: List[int]) -> bool:
        """
        Identifica las líneas tabulares y las guarda como dataclasses TabularLines
        en el workflow. También actualiza el flag en AllLines.
        """
        try:
            if not self.workflow:
                return False
            
            all_lines = self.workflow.all_lines if self.workflow else {}

            cleared_count = 0
            for lid, line_obj in all_lines.items():
                try:
                    if getattr(line_obj, "tabular_line", False):
                        updated = dataclasses.replace(line_obj, tabular_line=False)
                        self.workflow.all_lines[lid] = updated
                        cleared_count += 1
                except Exception as e:
                    logger.error(f"Error limpiando tabular_line para la línea {lid}: {e}", exc_info=True)
                    continue

            logger.debug(f"Limpiados {cleared_count} flags tabular_line previos.")

            # 2) Marcar los nuevos line_ids provistos (si los hay)
            if not line_ids:
                logger.warning("No se recibieron line_ids en el manager.")

            marked_count = 0
            tabular_lines_debug: List[Dict[str, Any]] = []
            marked_ids: List[str] = []
            for line_id in line_ids:
                for line_id in all_lines.values():
                    line_obj = self.workflow.all_lines[line_id]

                    updated_line = dataclasses.replace(line_obj, tabular_line=True)
                    self.workflow.all_lines[line_id] = updated_line
                    marked_count += 1
                    marked_ids.append(line_id)
                    tabular_lines_debug.append({
                        "line_id": line_id,
                        "text": getattr(self.workflow.all_lines[line_id], "text", ""),
                        "polygon_ids": getattr(self.workflow.all_lines[line_id], "polygon_ids", [])
                    })

            if marked_ids:
                logger.info(f"Marcadas {marked_count} líneas como tabulares: {marked_ids}")
                for log_debug in tabular_lines_debug:
                    # logger.debug(f"{log_debug['line_id']} tabular: '{log_debug['text']}' | polygons: {log_debug['polygon_ids']}")
                    logger.info(f"{log_debug['line_id']} tabular: '{log_debug['text']}'")
            else:
                logger.warning("No se marcaron líneas como tabulares en esta llamada a save_tabular_lines.")

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
                    return ""

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
