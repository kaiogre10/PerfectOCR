# core/domain/data_formatter.py
from domain.data_models import WorkflowData, StructuredData, Metadata, Polygons, CroppedImage, AllLines, FullImage, Payload
import numpy as np
import dataclasses
import logging
from typing import Dict, Any, Optional, List, Tuple
from utils.image_utils import normalice_image
import pandas as pd #type: ignore
from services.log_service import get_caller_info
from domain.class_models import SemantiClass

logger = logging.getLogger(__name__)

class DataFormatter:
    """Válvula de entrada/salida para todas las operaciones del workflow."""
    __slots__ = ("workflow", "payload", "text_ocr_log", "key_fields_log", "kf_list_log", "lines_log", "table_lines_log", "table_correct_log")
    def __init__(self, logs_config: Dict[str, Any]):
        self.workflow: Optional[WorkflowData] = None
        self.payload: Optional[Payload] = None
        
        self.text_ocr_log = logs_config.get("text_ocr", False)
        self.key_fields_log = logs_config.get("key_fields", False)
        self.kf_list_log = logs_config.get("kf_list_log", [])
        self.lines_log = logs_config.get("lines", False)
        self.table_lines_log = logs_config.get("table_lines", False)
        self.table_correct_log = logs_config.get("table_correct", False)

    def reset_data(self) -> None:
        self.workflow: Optional[WorkflowData] = None
        self.payload: Optional[Payload] = None
        
    def create_workflow(self, gray_img: np.ndarray[Any, np.dtype[np.uint8]], metadata: Dict[str, Any]) -> bool:
        """Crea un nuevo workflow usando dataclasses"""
        try:
            image_name=str(metadata.get("image_name", ""))
            metadata_obj = Metadata(
                image_name=image_name,
                dpi=int(metadata.get("dpi", 0)),
                img_dims = (0 , 0)
            )
            full_image = FullImage(full_img = gray_img)
            self.workflow = WorkflowData(
                full_img=full_image,
                metadata=metadata_obj,
                polygons=None,
                all_lines=None,
                table_data=None,
            )
            logger.debug(f"WORKFLOWDICT DREADO ÉXITOSAMENTE: '{image_name}'")
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
                poly_index = poly_data["poly_index"]
                coords = poly_data["polygon_coords"]
                bbox = poly_data["bounding_box"]
                centroid = poly_data["centroid"]

                # Crear objeto Polygons y agregar al diccionario
                polygon_obj = Polygons(
                    polygon_id=poly_id,
                    poly_index=poly_index,
                    polygon_coords = coords,
                    bounding_box = bbox,
                    centroid = centroid,
                    cropped_img=None,
                    ocr_text=None,
                    key_field=None,
                    semantic_clasification=[SemantiClass.DESCRIPTIVE],
                    cuant_chars=0
                )
                polygons_dataclass[poly_id] = polygon_obj
                                
            if self.workflow:
                self.workflow.polygons = polygons_dataclass
                
            logger.debug(f"Polígonos creados y validados: {len(polygons_dataclass)}")
            return True
            
        except Exception as e:
            logger.error(f"Error en create_polygon_dicts: {e}", exc_info=True)
        return False
            
    def get_full_img(self) -> Optional[FullImage]:
        return self.workflow.full_img if self.workflow else None
        
    def delete_cropped_images(self):
        """Libera todas las imágenes recortadas de los polígonos para ahorrar memoria."""
        if not self.workflow or not self.workflow.polygons:
            raise RuntimeError("No hay workflow inicializado para limpiar imágenes recortadas.")
            
        for poly_id, polygon in self.workflow.polygons.items():
            polygon.cropped_img = None
            updated_polygon = dataclasses.replace(polygon, cropped_img=None)
            self.workflow.polygons[poly_id] = updated_polygon
        logger.debug("Todas las imágenes recortadas han sido liberadas de memoria")

    def update_full_img(self, corrected: bool, full_img: Optional[np.ndarray[Any, np.dtype[np.uint8]]]=None) -> bool:
        """Actualiza o vacía la imagen completa en el workflow"""
        try:
            if not self.workflow:
                logger.error("No hay workflow inicializado para actualizar full_img.")
                return False
                
            if full_img is None and not corrected:
                # Medir dims de la imagen real almacenada antes de liberar memoria
                self.workflow.full_img = None
                self.workflow = dataclasses.replace(self.workflow, full_img=None)
                logger.debug("Full image liberada")
                return True
            
            if corrected or full_img is not None:
                # Normalizar si se recibe la dataclass FullImage corregida
                img_arr = normalice_image(full_img)
                if img_arr is None:
                    logger.critical(f"Error normalizando")
                    return False
                # Wrap en la dataclass FullImage y actualizar workflow
                full_image_obj = FullImage(img_arr)
                self.workflow = dataclasses.replace(self.workflow, full_img=full_image_obj)
                
                up_img = self.workflow.full_img.full_img if self.workflow else None # type: ignore
                if up_img is None:
                    return True
                h = up_img.shape[0]
                w = up_img.shape[1]
                dims = (h, w)
                self.workflow.metadata.img_dims = dims # type: ignore

                logger.debug("Imagen actualizada con éxito.")
                return True
            return True
            
        except Exception as e:
            logger.error(f"Error actualizando full_img: {e}", exc_info=True)
            return False
                        
    def save_cropped_images(self, cropped_images: Dict[str, np.ndarray[Any, np.dtype[np.uint8]]]) -> bool:
        """Guarda imágenes recortadas y geometría de recorte en los polígonos de las dataclasses"""
        try:
            if not self.workflow or not self.workflow.polygons:
                logger.error("No hay workflow inicializado para guardar imágenes recortadas.")
                return False

            total_img = len(cropped_images)

            logger.debug(f"'{total_img}' imágenes recortadas recibidas para guardar.")

            for poly_id, img in cropped_images.items():
                if poly_id in self.workflow.polygons:
                    polygon = self.workflow.polygons[poly_id]

                    cropped_image_obj = CroppedImage(img)
                    # Crear nuevo polígono con la imagen recortada y la geometría
                    updated_polygon = dataclasses.replace(polygon, cropped_img=cropped_image_obj)
                    self.workflow.polygons[poly_id] = updated_polygon

            logger.debug(f"Guardadas {len(cropped_images)} imágenes recortadas y geometría de recorte")
            return True
        except Exception as e:
            logger.error(f"Error guardando imágenes recortadas y geometría: {e}", exc_info=True)
        return False
                    
    def update_ocr_results(self, final_results: Dict[str, Dict[str, Any]]) -> bool:
        """Actualiza los resultados de OCR en las dataclasses de polígonos."""
        worker, _ = get_caller_info()
        try:
            if not self.workflow or not self.workflow.polygons:
                logger.error("No hay workflow inicializado para actualizar resultados OCR.")
                return False
                                    
            if not final_results:
                logger.error(f"No hay Texto OCR")
                return False
                
            reindexed_polygons: Dict[str, Polygons] = {}
            new_id = 0
            for i, (poly_id, res), in enumerate(final_results.items()):
                if poly_id in self.workflow.polygons:
                    polygon = self.workflow.polygons[poly_id]
                    text=res.get("text", "")
                    
                    if not text or text is None:
                        continue
                        
                    cuant_c=polygon.cuant_chars if not res.get("cuant_chars") else 0
                    sc = polygon.semantic_clasification if not res.get("sc") else [SemantiClass.DESCRIPTIVE]
                    
                    new_id += 1
                    new_idx = f"poly_{new_id:04d}"
                    updated_polygon = dataclasses.replace(
                        polygon,
                        polygon_id=new_idx,
                        poly_index=new_id,
                        ocr_text=text,
                        semantic_clasification=sc,
                        cuant_chars=cuant_c
                    )
                    reindexed_polygons[new_idx] = updated_polygon
                else:
                    logger.warning(f"Polígono {poly_id} no encontrado en workflow polygons")
            self.workflow.polygons = reindexed_polygons

            if self.text_ocr_log and (worker == "PaddleOCRWrapper" or worker == "paddle_wrapper"):
                polys = self.workflow.polygons if self.workflow else None
                if polys:
                    
                    logger.info("------TEXTO OCR RAW------")
                    for pid, poly, in polys.items():
                        logger.info(f"{pid}: '{poly.ocr_text}'")
                    logger.info("------FIN DEL TEXTO OCR RAW------")
            return True
        except Exception as e:
            logger.error(f"Error actualizando resultados OCR: {e}", exc_info=True)
        return False

    def update_semantic_clasification(self, final_results: Dict[str, Tuple[List[int], int]]) -> bool:
        """Actualiza el semantic_clasification de los polígonos."""
        try:
            if not self.workflow or not self.workflow.polygons:
                logger.error("No hay workflow inicializado para actualizar resultados OCR.")
                return False

            for poly_id, semantic_type in final_results.items():
                if poly_id in self.workflow.polygons:
                    polygon = self.workflow.polygons[poly_id]
                    # Actualizar semantic_clasification
                    updated_polygon = dataclasses.replace(polygon, semantic_clasification=semantic_type[0], cuant_chars=semantic_type[1])
                    self.workflow.polygons[poly_id] = updated_polygon
                    
            return True
            
        except Exception as e:
            logger.error(f"Error actualizando múltiples polígonos: {e}", exc_info=True)
        return False
                
    def update_key_field(self, polygon_updates: Optional[Dict[str, List[int]]]) -> bool:
        """Actualiza los datos de los polígonos en las dataclasses de polígonos."""
        try:
            if not self.workflow or not self.workflow.polygons:
                logger.error("No hay workflow inicializado para actualizar polígonos.")
                return False
            
            if not polygon_updates:
                logger.warning("Sin Key_fields")
                return True

            updated_count = 0

            for poly_id, key_field in polygon_updates.items():
                if poly_id in self.workflow.polygons:
                    polygon = self.workflow.polygons[poly_id]
                    updated_polygon = dataclasses.replace(polygon, key_field=key_field, semantic_clasification=[SemantiClass.UNIQUE], cuant_chars=0)
                    self.workflow.polygons[poly_id] = updated_polygon
                    updated_count += 1
                    
            if updated_count > 0:
                if self.key_fields_log:
                    for pid, poly_data in self.workflow.polygons.items():
                        kf = poly_data.key_field or None
                        if kf is not None:
                            if any(k in self.kf_list_log for k in kf):
                                logger.info(f"UPDATED: {pid}, key_field: {kf}, text: '{poly_data.ocr_text}'")
                                
                    logger.info(f"Actualizados {updated_count} polígonos con key_fields")
            else:
                logger.warning("No hubo poligonos con key_field")
                return True
            return True
            
        except Exception as e:
            logger.warning(f"Error actualizando múltiples polígonos: {e}", exc_info=True)
        return True

    def create_text_lines(self, lines_info: Dict[str, Any]) -> bool:
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
                all_lines_dataclasses[line_id] = AllLines(
                    lineal_id=line_id,
                    line_index=line_data.get("line_index"),
                    text=line_data.get("text", ""),
                    polygon_ids=line_data["polygon_ids"],
                    polygons_index=line_data["polygons_index"],
                    line_centroid=line_data["line_centroid"] or [0.0, 0.0],
                    line_bbox=line_data["line_bbox"] or [0.0, 0.0, 0.0, 0.0],
                    tabular_line=line_data["tabular_line"],
                    header_line=line_data["header_line"] or None,
                    footer_line=line_data["footer_line"] or None,
                    t_cuant = line_data["t_cuant"]
                )
            
                self.workflow.all_lines = all_lines_dataclasses
                
            if self.lines_log:
                all_lines = self.workflow.all_lines if self.workflow else None
                if all_lines:
                    for lid, l in all_lines.items():
                        line_text= l.text
                        # line_semantic = fast_classfier(line_text)
                        logger.info(f"{lid}: '{line_text}'")
            return True
                        
        except ValueError as e:
            logger.error(f"Error guardando líneas de texto: {e}", exc_info=True)
        return False

    def save_tabular_lines(self, line_ids: List[str]) -> bool:
        """Identifica las líneas tabulares"""
        try:
            if not self.workflow or not self.workflow.all_lines:
                return False
            
            all_lines = self.workflow.all_lines
            for lid, line_obj in all_lines.items():
                if line_obj.tabular_line:
                    updated = dataclasses.replace(line_obj, tabular_line=False)
                    self.workflow.all_lines[lid] = updated
            # 2) Marcar los nuevos line_ids provistos (si los hay)
            if not line_ids:
                logger.warning("No se recibieron line_ids en el manager.")

            marked_count = 0
            tabular_lines_debug: List[Dict[str, Any]] = []
            marked_ids: List[str] = []
            for line_id in line_ids:
                if line_id in all_lines: 
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
                else:
                    logger.warning(f"line_id '{line_id}' no encontrado en all_lines")

            if marked_ids:
                if self.table_lines_log:
                    for log_debug in tabular_lines_debug:
                        logger.info(f"{log_debug['line_id']} tabular: '{log_debug['text']}'")
                    logger.info(f"Marcadas {marked_count} líneas como tabulares")
                return True
            else:
                logger.warning("No se marcaron líneas como tabulares.")
                return False
                
        except Exception as e:
            logger.error(f"Error marcando líneas como tabulares: {e}", exc_info=True)
        return False
                
    def save_final_output(self, df: pd.DataFrame, key_data: Dict[str, str]) -> bool:
        """Actualiza los datos finales"""
        try:
            if not self.workflow:
                return False

            if df.empty and not key_data:
                updated_data = StructuredData(df_table=None, global_data={})
                self.workflow.table_data = dataclasses.replace(updated_data)
                logger.error("Sin DATA FRAME NI DATOS")
                return False
            
            if not df.empty and key_data:
                data_obj = StructuredData(df_table=df, global_data=key_data)
                self.workflow.table_data = dataclasses.replace(data_obj)
                # logger.info("ACTUALIZADOS AMBOS CAMPOS")

                if self.table_correct_log:
                    table_f = self.workflow.table_data.df_table
                    global_data = self.workflow.table_data.global_data
                    logger.debug("Tabla recibida:\n"f"{table_f.to_string(index=True)}" # type: ignore
                    "\n"f"GLOBAL_DATA:\n"f"{global_data}")
                return True
            
            df_tab = self.workflow.table_data.df_table if self.workflow.table_data else pd.DataFrame()
            glob_data = self.workflow.table_data.global_data if self.workflow.table_data else {}

            if key_data and df.empty:
                updated_data = StructuredData(df_tab, key_data)
                self.workflow.table_data = dataclasses.replace(updated_data)
                # logger.info("ACTUALIZADA INFORMACIÓN GLOBAL")
                return True

            if not df.empty and not key_data:
                updated_data = StructuredData(df, glob_data)
                self.workflow.table_data = dataclasses.replace(updated_data)
                # logger.info("ACTUALIZADO DF")
                return True
            
        except Exception as e:
            logger.error(f"Error guardando structured_table en memoria: {e}", exc_info=True)
        return False
    
    def store_payload(self, payloads: List[str]):
        payload = payloads[0]
        image_name = payloads[1]
        payload = Payload(payload=payload, name=image_name)
        self.payload = dataclasses.replace(payload)
        return True
