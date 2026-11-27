# domain/data_formatter.py
import logging
import sys
from bidings import arrays, image
from domain.data_models import WorkflowData, StructuredData, Polygons, AllLines, Payload
from domain.buffer_models import FullImageKey, BoundingBoxesKey, CentroidsKey, CroppedBufferKey, AoSCroppedImageKey
import numpy as np
import dataclasses
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd # type: ignore
from services.log_service import get_caller_info
from domain.class_models import SemantiClass
from domain.buffer_handler import ImageBuffer, ArrayBuffer
from utils.text_utils import set_ids

image_type = b"uint8"

logger = logging.getLogger(__name__)

class DataFormatter:
    """Válvula de entrada/salida para todas las operaciones del workflow."""
    __slots__ = ("workflow", "payload", "text_ocr_log", "key_fields_log", "kf_list_log", "lines_log", "table_lines_log", "table_correct_log", "refined_text")
    def __init__(self, logs_config: Dict[str, Any]):
        self.workflow: Optional[WorkflowData] = None
        self.payload: Optional[Payload] = None
        
        self.text_ocr_log = logs_config.get("text_ocr", False)
        self.key_fields_log = logs_config.get("text_keyfields", False)
        self.refined_text = logs_config.get("text_refined")
        self.kf_list_log = logs_config["kf_list_log"]
        self.lines_log = logs_config.get("lines", False)
        self.table_lines_log = logs_config.get("table_lines", False)
        self.table_correct_log = logs_config.get("table_correct", False)

    def reset_data(self) -> None:
        self.workflow = None
        self.payload = None
        
    def create_workflow(self, img_ptr: int, image_name: str) -> bool:
        """Crea un nuevo Workflow"""
        if not self.create_image(img_ptr):
            raise RuntimeError("ERROR ALMACENANDO IMAGEN EN MEMORIA")

        try:
            ptr = FullImageKey.img_ptr
            if not isinstance(ptr, int): # type: ignore
                return False
            
            full_img = ImageBuffer(ptr, image_type)
            
            # _dtype = full_img.dtype
            # _rows = full_img.rows
            # itemsize = np.dtype(_dtype).itemsize
            # bytes_per_row = _rows * itemsize
            
            # # Operación de redondeo al siguiente múltiplo de 32 mediante división entera
            # stride_bytes = ((bytes_per_row + 31) // 32) * 32

            # logger.info(f"DTYPE: {_dtype}, SHAPE: {full_img.shape}, STRIDES: {stride_bytes}")
        except RuntimeError as e:
            logger.error(f"ERROR ASIGNANDO LAS DIMENSIONES DE LA IMAGEN: {e}", exc_info=True)
            return False

        try:
            img_dims = full_img.shape
            self.workflow = WorkflowData(
                image_name=image_name,
                img_dims=img_dims,
                polygons=None,
                all_lines=None,
                table_data=None,
            )
            logger.debug(f"WORKFLOWDICT DREADO ÉXITOSAMENTE: '{image_name}'")
            return True

        except SystemError as e:
            logger.error(f"No se pudo crear el workflowDict: {e}", exc_info=True)
            return False
    
    def create_polygon_dicts(self, results: List[int]) -> bool:
        try:
            if self.workflow is None:
                return False
            
            polygons_dataclass: Dict[str, Polygons] = {}
            
            for idx, croped_ptr in enumerate(results):
                poly_id = set_ids(idx)
                # bbox = poly_data["bounding_box"]
                # centroid = poly_data["centroid"]

                # Crear objeto Polygons y agregar al diccionario
                polygon_obj = Polygons(
                    polygon_id=poly_id,
                    poly_index=idx,
                    cropped_img=croped_ptr,
                    # bounding_box=bbox,
                    # centroid=centroid,
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

    def create_polygons(self, polygons_detected: List[int]) -> bool:    
        try:
            BoundingBoxesKey(bboxes_ptr=polygons_detected[0])
            bboxes_ptr = BoundingBoxesKey.bboxes_ptr
            if not bboxes_ptr:
                raise TypeError("NO SE DEVOLVIO PUNTERO A BBOXES")
            
        except TypeError as e:
            logger.exception(f"No se pudo almacenar el puntero de bboxes: {e}", exc_info=True)
            return False
        try:
            CentroidsKey(polygons_detected[1])
            centroids_ptr = CentroidsKey.centroids_ptr
            if not centroids_ptr:
                raise TypeError("NO SE DEVOLVIO PUNTERO A CENTROIDES")
        
        except TypeError as e:
            logger.exception(f"No se pudo almacenar el puntero a centroides: {e}", exc_info=True)
            return False
        try:
            CroppedBufferKey(polygons_detected[2])
            cropped_buffer_ptr = CroppedBufferKey.cropped_buffer_ptr
            if not cropped_buffer_ptr:
                raise TypeError("NO SE DEVOLVIO PUNTERO A BUFFER DE CROPS")

        except TypeError as e:
            logger.exception(f"No se pudo almacenar el puntero a BUFFER DE CROPS: {e}", exc_info=True)
            return False

        try:
            AoSCroppedImageKey(polygons_detected[3])
            cropped_aos_ptr = AoSCroppedImageKey.cropped_aos_ptr
            if not cropped_aos_ptr:
                raise TypeError("NO SE DEVOLVIO PUNTERO A AoS")

        except TypeError as e:
            logger.exception(f"No se pudo almacenar el puntero a AoS: {e}", exc_info=True)
            return False

        total_bboxes = ArrayBuffer(bboxes_ptr).shape[0]
        total_centroids = ArrayBuffer(centroids_ptr).shape[0]

        if total_bboxes != total_centroids:
            logger.error(f"DISPARIDAD EN LA GEOMETRIA: BBOXES: {total_bboxes} != {total_centroids} CENTROIDS")
            return False

        list_ptr = image.get_crops_ptr(cropped_aos_ptr, total_bboxes)
        total_crops = len(list_ptr)

        if not list_ptr or total_crops < 1:
            logger.error(f"ERRIR VALIDANDO CROPS: {total_crops}")
            return False

        if total_crops != total_centroids:
            logger.error(f"DISPARIDAD EN LOS CROPS: {total_crops}")
            return False

        if not self.create_polygon_dicts(list_ptr):
            logger.error("NO SE PUDIERON CREAR LOS POLÍGONOS")
            return False
        
        polygon = self.workflow.polygons if self.workflow else {}
        if not polygon:
            logger.error(f"NO SE GUARDARON LOS POLÍGONONS")
            return False
        
        total_polygons = len(polygon)
        if total_bboxes != total_polygons:
            logger.error(f"INCONGRUENCIA EN CANTIDAD DE POLIGONOS Y BBOXES GENERADOS")
            return False

        logger.warning(f"RECORTADOS: {total_crops} POLIGONOS, {total_bboxes} BBOXES, {total_centroids} CENTROIDS")
        return True

    def create_image(self, img_ptr: Any) -> bool:
        try:
            FullImageKey(img_ptr=img_ptr)
        except TypeError as e:
            logger.error(f"No se pudo almacenar el puntero: {e}", exc_info=True)
            return False
        
        logger.warning(f"PUNTERO ALMACENADO: '{img_ptr}'")
        return True

    def get_array_buffer(self, rows: int, cols: int, channels: int, dtype: bytes) -> Optional[int]:
        """Solicita un buffer de memoria a C. Devuelve puntero"""
        worker, line = get_caller_info()
        if not dtype:
            raise TypeError("NO SE INGRESO TYPO DE ARRAY")
        
        arr_ptr = arrays.create_new_array(rows, cols, channels, dtype)
        logger.warning(f"'{worker}'-{line}  SOLICITO: '{arr_ptr}'")
        if not arr_ptr or not isinstance(arr_ptr, int): # type: ignore
            raise RuntimeError("POSIBLE MEMMORY LEAK")
            
        return arr_ptr

    def get_img_ptr(self) -> int:
        img_ptr = FullImageKey.img_ptr
        if not img_ptr:
            return -1
        else:
            return img_ptr

    def get_bboxes_ptr(self) -> int:
        bboxes_ptr = BoundingBoxesKey.bboxes_ptr
        if not bboxes_ptr:
            return -1
        else:
            return bboxes_ptr

    def get_all_cropped_img_ptr(self) -> int:
        cropped_buffer_ptr = CroppedBufferKey.cropped_buffer_ptr
        if not cropped_buffer_ptr:
            return -1
        else:
            return cropped_buffer_ptr

    def get_img_copy(self) -> Tuple[Optional[int], Optional[np.ndarray[Any, np.dtype[np.uint8]]]]:
        """Devuelve puntero[0] y copia de la imagen principal[1] para futuras liberaciones"""
        worker, line = get_caller_info()
        ptr = self.get_img_ptr()
        copy_ptr = image.copy_image(ptr, image_type)
        if not copy_ptr:
            return None, None
        
        logger.warning(f"'{worker}'-{line}  SOLICITO: '{copy_ptr}'")
        full_img = ImageBuffer(copy_ptr, image_type).data
        return copy_ptr, full_img

    def delete_buffers(self, ptr_list: List[int]) -> bool:
        """Centraliza la liberación de varios punteros"""
        worker, _ = get_caller_info()
        for ptr in ptr_list:
            if ptr < 1:
                logger.warning(f"'{worker}' ENTREGO PUNTERO INVALIDO: {ptr}")

            if not self.free_buffer(ptr):
                raise BufferError(f"'{worker}' ENTREGO PUNTERO CON SEGMENT FAULT: {ptr}")
            
            logger.info(f"'{worker}' LIBERO PUNTERO: {ptr}")
            
        return True

    def free_buffer(self, copy_ptr: int):
        worker, line = get_caller_info()
        if copy_ptr:
            try:
                image.release_img(ptr_addr=copy_ptr)
            except RuntimeError as e:
                logger.exception(f"ERROR DE '{worker}' LINE: {line} LIBERANDO COPIA: {e}", exc_info=True)
                return False
            
            if worker != "data_formatter.py":
                logger.warning(f"'{worker}' LIBERO COPIA {copy_ptr} en linea: '{line}'")
            copy_ptr = None
            del copy_ptr
            return True

        else:            
            ptr = FullImageKey.img_ptr
    
            if not ptr:
                logger.warning(f"'{worker}' INTENTO LIBERAR: '{ptr}' PERO LIBERADO ANTERIORMENTE")
                return True

            try:
                FullImageKey.clear_img()
                
            except RuntimeError as e:
                logger.error(f"ERROR DE '{worker}' ELIMINANDO LA FULL IMG: {e}", exc_info=True)
                return False

            if worker != "data_formatter.py":
                logger.warning(f"'{worker}' linea '{line}': LIBERO IMAGEN")
                
            return True
                    
    def get_full_img(self) -> Optional[np.ndarray[Any, np.dtype[np.uint8]]]:        
        ptr = FullImageKey.img_ptr
        full_img = ImageBuffer(ptr, image_type)
        return full_img.data

    def update_image(self, new_ptr: int) -> bool:
        worker, _ = get_caller_info()
        try:
            logger.warning(f"'{worker}' LIBERO COPIA")

            if not self.create_image(new_ptr):
                raise ReferenceError(f"ERROR CREANDO BUFFER ACTUALIZADO: {new_ptr}")
            
        except Exception as e:
            logger.error(f"ERROR ACTUALIZANDO LA IMAGEN, {e}", exc_info=True)
            return False

        return True

    def delete_cropped_images(self):
        """Libera todas las imágenes recortadas de los polígonos para ahorrar memoria."""
        if not self.workflow or not self.workflow.polygons:
            raise RuntimeError("No hay workflow inicializado para limpiar imágenes recortadas.")
        polygons = self.workflow.polygons
        totak_polys = len(polygons)
        cropped_buffer_ptr = CroppedBufferKey.cropped_buffer_ptr
        cropped_aos_ptr = AoSCroppedImageKey.cropped_aos_ptr
        try:

            image.release_crops(cropped_buffer_ptr, cropped_aos_ptr, totak_polys)

            # CroppedBufferKey.free_cropped_buffer_images()
            # AoSCroppedImageKey.free_cropped_asos()
        except RuntimeError as e:
            logger.exception(f"ERROR LIBERANDO PUNTEROS AOS: {e}", exc_info=True)
            sys.exit()

        logger.warning("EXITO LIBERANDO CROPPED_IMG")
        
        polygon_to_free: Dict[str, Polygons] = {}            
        for _, (poly_id, polygon) in enumerate(self.workflow.polygons.items()):
            cropped_img = polygon.cropped_img
            del cropped_img
            polygon_to_free[poly_id] = dataclasses.replace(polygon, cropped_img=None)

        self.workflow.polygons = polygon_to_free
        logger.debug("Todas las imágenes recortadas han sido liberadas de memoria")
                        
    def save_cropped_images(self, cropped_images: Dict[str, Any], reindex: bool) -> bool:
        """Guarda imágenes recortadas y geometría de recorte en los polígonos de las dataclasses"""
        try:
            if not self.workflow or not self.workflow.polygons:
                logger.error("No hay workflow inicializado para guardar imágenes recortadas.")
                return False
            
            logger.debug(f"'{len(cropped_images)}' imágenes recortadas recibidas para guardar.")
            reindexed_polygons: Dict[str, Polygons] = {}
            for i, (poly_id, crop) in enumerate(cropped_images.items()):

                polygon = self.workflow.polygons[poly_id]
                if not reindex:
                    new_idx = poly_id
                    updated_polygon = dataclasses.replace(polygon, cropped_img=crop)
                else:
                    new_idx = set_ids(i)
                    # Crear nuevo polígono con la imagen recortada
                    updated_polygon = dataclasses.replace(polygon, polygon_id=new_idx, poly_index=i, cropped_img=crop)

                reindexed_polygons[new_idx] = updated_polygon

            self.workflow.polygons = reindexed_polygons

            logger.debug(f"Guardadas imágenes recortadas y geometría de recorte")
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
            idx = 0
            
            for _, (poly_id, res), in enumerate(final_results.items()):
                polygon = self.workflow.polygons[poly_id]
                text=res.get("text", "")
                
                if not text:
                    continue
                    
                cuant_c = polygon.cuant_chars if not res.get("cuant_chars") else 0
                sc = polygon.semantic_clasification if not res.get("sc") else [SemantiClass.DESCRIPTIVE]
                
                new_idx = set_ids(idx)
                idx += 1
                updated_polygon = dataclasses.replace(
                    polygon,
                    polygon_id=new_idx,
                    poly_index=idx,
                    ocr_text=text,
                    semantic_clasification=sc,
                    cuant_chars=cuant_c
                )
                reindexed_polygons[new_idx] = updated_polygon

            self.workflow.polygons = reindexed_polygons

            if self.text_ocr_log and (worker == "PaddleOCRWrapper" or worker == "paddle_wrapper.py"):
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

            for _, (poly_id, semantic_type) in enumerate(final_results.items()):
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
                all_lines_dataclasses[line_id] = AllLines(lineal_id=line_id, **line_data)
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
    
    def store_payload(self, payload: str, buff_size: int):
        payload_obj = Payload(payload=payload, buff_size=buff_size)
        self.payload = dataclasses.replace(payload_obj)
        return True
