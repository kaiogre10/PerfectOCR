# core/domain/data_formatter.py
from core.domain.data_models import WORKFLOW_SCHEMA, WorkflowDict, DENSITY_ENCODER, StructuredTable, Geometry, Metadata, Polygons, CroppedGeometry, CroppedImage, AllLines, LineGeometry
import numpy as np
import time
import jsonschema
import logging
import json
from typing import Dict, Any, Optional, List, Tuple
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
            "all_lines": {}
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

    def update_full_img(self, full_img: (Optional[np.ndarray[Any, np.dtype[np.uint8]]])=None) -> bool:
        """Actualiza o vacía la imagen completa en el workflow"""
        try:
            if not self.workflow_dict:
                logger.error("No hay workflow inicializado para actualizar full_img.")
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

            logger.debug(f"Guardadas {len(cropped_images)} imágenes recortadas y geometría de recorte en dataclasses.")
            return True
        except Exception as e:
            logger.error(f"Error guardando imágenes recortadas y geometría: {e}")
            return False
        
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
                    tabular_line=False,
                    header_line=False 
                )
            
            # Actualiza la fuente de verdad (dataclasses)
            self.workflow.all_lines = all_lines_dataclasses
            
            num_lines = len(all_lines_dataclasses)
            logger.debug(f"Guardadas {num_lines} líneas reconstruidas en workflow_dict y dataclasses.")
            for line_id, line_data in self.workflow.all_lines.items():
                return True
        except Exception as e:
            logger.error(f"Error guardando líneas de texto: {e}", exc_info=True)
            return False        
            

    def save_tabular_lines(self, line_ids: List[str]) -> bool:
        """
        Identifica las líneas tabulares y las guarda como dataclasses TabularLines
        en el workflow. También actualiza el flag en AllLines.
        Además, loguea las líneas tabulares con su texto.
        """
        try:
            if not self.workflow or not line_ids:
                return False

            marked_count = 0
            tabular_lines_info = []
            for line_id in line_ids:
                if line_id in self.workflow.all_lines:
                    self.workflow.all_lines[line_id].tabular_line = True
                    marked_count += 1
                    # Guardar info para log
                    line_obj = self.workflow.all_lines[line_id]
                    tabular_lines_info.append({
                        "line_id": line_id,
                        "text": line_obj.text,
                        "polygon_ids": line_obj.polygon_ids
                    })

            logger.info(f"Marcadas {marked_count} líneas como tabulares")
            if tabular_lines_info:
                logger.info("Líneas tabulares detectadas (id, texto, encoded_text, polygon_ids):")
                for log_info in tabular_lines_info:
                    logger.info(f"  {log_info['line_id']}: '{log_info['text']}' | polygons: {log_info['polygon_ids']}")

            return marked_count > 0

        except Exception as e:
            logger.error(f"Error marcando líneas como tabulares: {e}")
            return False
                
    def save_structured_table(self, df: pd.DataFrame, columns: List[str], semantic_types: Optional[List[str]] = None) -> bool:
        try:
            self.structured_table = StructuredTable(df=df, columns=columns, semantic_types=semantic_types)
            return True
        except Exception as e:
            logger.error(f"Error guardando structured_table en memoria: {e}")
            return False

    def update_lines_metadata(self, updates: List[Tuple[str, Dict[str, Any]]]):
        """Actualiza las líneas (AllLines y TabularLines) con nuevos campos de metadatos."""
        if not self.workflow:
            return
            
        for line_id, data_to_update in updates:
            # También es buena idea marcar la línea en la lista global (AllLines)
            if line_id in self.workflow.all_lines:
                line_obj = self.workflow.all_lines[line_id]
                # Marcarla como header si la clave existe en la actualización
                if 'header_line' in data_to_update:
                    line_obj.header_line = data_to_update['header_line']

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

    def _row_to_detalle(self, row: List[str]) -> Dict[str, Any]:
    # Mapea columnas esperadas al esquema de DetallesCompra
        return {
            "IDDetalle": None,  # Se generará automáticamente en la BD
            "IDRegistro": None,  # Se vinculará con el registro principal
            "Cantidad": self._parse_number(row.get("cantidad") or row.get("c")),
            "SKU": row.get("sku"),
            "ProductoEstandarizado": str(row.get("descripcion") or row.get("desc") or "").strip(),
            "PrecioUnitario": self._parse_number(row.get("precio_unitario") or row.get("pu")),
            "ImporteRaw": self._parse_number(row.get("precio_total") or row.get("mtl")),
            "ImporteCalculado": self._parse_number(row.get("precio_total") or row.get("mtl"))
        }
    
    def to_db_payload(self) -> dict:
        """
        Construye un payload JSON-serializable:
        { registro: {...}, detalles: [...], provenance: {...}, raw_table: [...] }
        """
        if not self.workflow:
            return {}

        wf = self.workflow
        md = wf.metadata if hasattr(wf, "metadata") else (wf.get("metadata") if isinstance(wf, dict) else {})
        dict_id = getattr(wf, "dict_id", None) or (md.get("dict_id") if isinstance(md, dict) else None) or f"UNK-{int(time.time()*1000)}"

        registro = {
            "IDRegistro": dict_id,
            "FolioDocumento": md.get("FolioDocumento") if isinstance(md, dict) else getattr(md, "FolioDocumento", None),
            "FechaDocumento": md.get("FechaDocumento") if isinstance(md, dict) else getattr(md, "FechaDocumento", None),
            "ProveedorEstandarizado": md.get("ProveedorEstandarizado"),
            "RFCProveedor": md.get("RFCProveedor") if isinstance(md, dict) else getattr(md, "RFCProveedor", None),
            "MontoTotalDocumento": self._parse_number(md.get("MontoTotalDocumento") if isinstance(md, dict) else None),
            "TipoDocumento": md.get("TipoDocumento") if isinstance(md, dict) else getattr(md, "TipoDocumento", None),
            "FechaDigitalizacion": time.strftime("%Y%m%d%H%M%S")
        }

        detalles = []
        # si tienes structured_table la usamos
        if self.structured_table and hasattr(self.structured_table, "df"):
            df = self.structured_table.df
            cols = self.structured_table.columns if self.structured_table.columns else list(df.columns)
            for _, r in df.iterrows():
                row = {}
                # mapear por índice de columnas a col_0..col_n para _row_to_detalle
                for i, c in enumerate(cols):
                    row[f"col_{i}"] = r.get(c) if c in df.columns else r[i]
                detalles.append(self._row_to_detalle(row))
        else:
            # fallback: usar polygons/text lines como detalle descriptivo
            polygons = getattr(wf, "polygons", {}) or {}
            for pid, p in (polygons.items() if isinstance(polygons, dict) else []):
                text = getattr(p, "ocr_text", None) if hasattr(p, "ocr_text") else (p.get("ocr_text") if isinstance(p, dict) else "")
                detalles.append({
                    "cantidad": None, "descripcion": text, "precio_unitario": None, "precio_total": None, "unidad": None, "sku": None
                })

        provenance = {
            "dict_id": dict_id,
            "image_name": md.get("image_name") if isinstance(md, dict) else getattr(md, "image_name", None),
            "formatter_version": "v1",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")
        }

        payload = {"registro": registro, "detalles": detalles, "provenance": provenance}
        # opcional: raw_table para auditoría
        if self.structured_table and hasattr(self.structured_table, "df"):
            payload["raw_table"] = {"columns": list(self.structured_table.df.columns), "rows": self.structured_table.df.fillna("").astype(str).values.tolist()}

        return payload

    def export_payload_json(self, path: str) -> bool:
        """Escribe el payload en disco para auditoría/revisión manual"""
        try:
            payload = self.to_db_payload()
            if not payload:
                return False
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error exportando payload json: {e}")
            return False
