# core/domain/data_formatter.py
from core.utils.data_utils import  DENSITY_ENCODER, CHAR_FRECUENCY, VECTOR_MEDIAN_DUMMIE, VECTOR_MEAN_DUMMIE, FRECUENCY_ENCODER
from core.domain.data_models import WorkflowDict, StructuredTable, Geometry, Metadata, Polygons, CroppedGeometry, CroppedImage, AllLines, LineGeometry, FullImage, SemanticClassification
import numpy as np
import dataclasses
import logging
import json
import time
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from core.utils.image_normalicer import normalice_image
import pandas as pd #type: ignore

logger = logging.getLogger(__name__)

class DataFormatter:
    """
    Válvula de entrada/salida para todas las operaciones del workflow.
    Los workers NO tocan directamente el workflow, solo pasan por aquí.
    """
    def __init__(self):
        self.workflow: Optional[WorkflowDict] = None
        self.encoder: Optional[Dict[str, float]] = None
        self.frecuency: Optional[Dict[str, float]] = None
        self.frecuency_encoder: Optional[Dict[str, float]] = None
        self.mean_dummie: Optional[Dict[str, float]] = None
        self.median_dummie: Optional[Dict[str, float]] = None
        self.structured_table: Optional[StructuredTable] = None
    
    def create_workflow(self, IDRegistro: str, gray_img: np.ndarray[Any, np.dtype[np.uint8]], metadata: Dict[str, Any]) -> bool:
        """Crea un nuevo workflow usando solo dataclasses"""
        
        try:
            full_img = normalice_image(gray_img)
            
            full_image = FullImage(
                full_img=(full_img)
                )
            
            metadata_obj = Metadata(
                image_name=str(metadata.get("image_name", "")),
                img_dims={
                    "width": int(metadata.get("img_dims", {}).get("width") or 0),
                    "height": int(metadata.get("img_dims", {}).get("height") or 0),
                    "size": int(metadata.get("img_dims", {}).get("size") or 0),
                },
                date_creation=str(metadata.get("date_creation" or "")),
            )

            self.workflow = WorkflowDict(
                IDRegistro=IDRegistro,
                full_img=full_image,
                metadata=metadata_obj,
                polygons={},
                all_lines={},
            )
            return True
            
        except Exception as e:
            logger.error(f"No se pudo crear el workflowDict: {e}", exc_info=True)
        return False
    
    def create_polygon_dicts(self, results: Optional[List[Any]]) -> bool:
        """Refactorizado para usar validación + dataclasses"""
        try:
            if self.workflow is None:
                return False
                
            if results is None:
                return False
            
            polygons_dataclass: Dict[str, Polygons] = {}
            
            for idx, poly_pts in enumerate(results[0]):
                poly_id = f"poly_{idx:04d}"
                
                # Cálculos vectorizados
                coords = np.array([[float(p[0]), float(p[1])] for p in poly_pts])
                bbox = np.array([coords[:, 0].min(), coords[:, 1].min(), 
                            coords[:, 0].max(), coords[:, 1].max()])
                centroid = coords.mean(axis=0)

                # Crear objeto Geometry
                geometry = Geometry(
                    polygon_coords=coords,
                    bounding_box=bbox,
                    centroid=centroid
                )

                # Crear objeto Polygons y agregar al diccionario
                polygon_obj = Polygons(
                    polygon_id=poly_id,
                    geometry=geometry,
                    cropedd_geometry=None, #type:ignore
                    cropped_img=None,
                    perimeter=None,
                    ocr_text=None,
                    ocr_confidence=None,
                    was_refined=False,
                    key_field=None,
                    semantic_clasification=None #type:ignore
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
            
    def get_frecuency_char(self) -> Dict[str, float]:
        """Obtiene los valores de frecuencia para letras"""
        try:
            if self.frecuency is None:
                self.frecuency = CHAR_FRECUENCY
            return self.frecuency
        
        except Exception as e:
            logger.warning(f"Error entregando frecuencias: {e}", exc_info=True)
            return {}
            
    def get_frecuency_encoder(self) -> Dict[str, float]:
        """
        Obtiene los valores de densos frecuencia para letras.
        """
        try:
            if self.frecuency_encoder is None:
                self.frecuency_encoder = FRECUENCY_ENCODER
            return self.frecuency_encoder
        
        except Exception as e:
            logger.warning(f"Error entregando frecuencias: {e}", exc_info=True)
            return {}

    def get_density_encoder(self) -> Dict[str, float]:
        """
        Obtiene los valores de densidad para letras.
        """
        try:
            if self.encoder is None:
                self.encoder = DENSITY_ENCODER
            return self.encoder
        
        except Exception as e:
            logger.warning(f"Error entregando frecuencias: {e}", exc_info=True)
            return {}

    def get_mean_dummie(self) -> Dict[str, float]:
        """
        Codifica líneas específicas usando DENSITY_ENCODER con operaciones optimizadas.
        Si no se especifican line_ids, codifica todas las líneas existentes.
        """
        try:
            if self.mean_dummie is None:
                self.mean_dummie = VECTOR_MEAN_DUMMIE
            return self.mean_dummie
        
        except Exception as e:
            logger.warning(f"Error entregando frecuencias: {e}", exc_info=True)
            return {}

    def get_median_dummie(self) -> Dict[str, float]:
        """
        Codifica líneas específicas usando DENSITY_ENCODER con operaciones optimizadas.
        Si no se especifican line_ids, codifica todas las líneas existentes.
        """
        try:
            if self.median_dummie is None:
                self.median_dummie = VECTOR_MEDIAN_DUMMIE
            return self.median_dummie
        
        except Exception as e:
            logger.warning(f"Error entregando frecuencias: {e}", exc_info=True)
            return {}

    def get_tabular_lines(self, return_objects: bool = False) -> Union[Dict[str, Any], List[str]]:
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

    def update_full_img(self, full_img: (Optional[np.ndarray[Any, np.dtype[np.uint8]]])=None) -> bool:
        """Actualiza o vacía la imagen completa en el workflow"""
        try:
            if not self.workflow:
                logger.error("No hay workflow inicializado para actualizar full_img.")
                return False
                
            if full_img is None:
                # Si se pasa None, vaciamos la imagen para liberar memoria
                self.workflow = dataclasses.replace(self.workflow, full_img=None)
                logger.debug(f"Imagen liberada con éxito: {full_img}")
                return True
            
            # Normalizar si se recibe la dataclass FullImage
            if isinstance(full_img, FullImage):
                img_arr = getattr(full_img, "full_img", None)
            else:
                img_arr = full_img
                                
            img_arr = normalice_image(full_img)
            
            # Wrap en la dataclass FullImage y actualizar workflow
            full_image_obj = FullImage(full_img=img_arr)
            self.workflow = dataclasses.replace(self.workflow, full_img=full_image_obj)
            logger.debug("Imagen actualizada con éxito.")
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
        
        min_threshold = 5
        max_threshold = 250
        white_poly_ids: List[str] = []
        
        # Detectar polígonos blancos/inválidos
        for poly_id, polygon in self.workflow.polygons.items():
            cropped_img = polygon.cropped_img.cropped_img if polygon.cropped_img else None
            
            if cropped_img is None or cropped_img.size == 0:
                white_poly_ids.append(poly_id)
                continue
            
            # Validación simple con .mean()
            img_mean = cropped_img.mean(dtype=int)
            if img_mean < min_threshold or img_mean > max_threshold:
                white_poly_ids.append(poly_id)
        
        # Eliminar polígonos blancos y reindexar
        if white_poly_ids:
            logger.info(f"Eliminando {len(white_poly_ids)} polígonos blancos/inválidos")
            
            # Eliminar polígonos blancos
            for poly_id in white_poly_ids:
                if poly_id in self.workflow.polygons:
                    del self.workflow.polygons[poly_id]
            
            # Reindexar polígonos restantes (patrón de poly_gone)
            remaining_polygons = list(self.workflow.polygons.items())
            new_polygons: Dict[str, Polygons] = {}
            
            for idx, (old_id, poly_obj) in enumerate(remaining_polygons): #type: ignore
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

            for poly_id, res in final_results.items():
                if poly_id in self.workflow.polygons:
                    polygon = self.workflow.polygons[poly_id]
                    updated_polygon = dataclasses.replace(
                        polygon,
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
                        
    def merge_semantics(self) -> bool:
        """
        Unifica los tipos semánticos en las dataclasses, convirtiendo
        todos los 'quantitative' a 'numeric'.
        """
        if not self.workflow or not self.workflow.polygons:
            return False

        updated_count = 0
        for poly_id, polygon in self.workflow.polygons.items():
            if polygon.semantic_clasification.quantitative:
                updated_polygon = dataclasses.replace(polygon, semantic_clasification='numeric')
                self.workflow.polygons[poly_id] = updated_polygon
                updated_count += 1

            if polygon.semantic_clasification.umd:
                updated_polygon = dataclasses.replace(polygon, semantic_clasification='code')
                self.workflow.polygons[poly_id] = updated_polygon
                updated_count += 1
        
        if updated_count > 0:
            logger.debug(f"Unificados {updated_count} polígonos de 'quantitative' a 'numeric'.")
        return True

    def create_semantic_clasification(self) -> bool:
        """
        Verifica si existe la clasificación semántica para los polígonos
        si no existe, crea todas False como fallback
        """
        try:
            if not self.workflow:
                logger.error("No hay workflow inicializado para actualizar resultados OCR.")
                return False
                
            polygons: Dict[str, Polygons] = self.workflow.polygons if self.workflow else {}
            created_count = 0
            
            for poly_id, poly_data in polygons.items():
                semantic_obj = poly_data.semantic_clasification
                if not semantic_obj:
                    semantic_obj = SemanticClassification(
                        numeric=False,
                        descriptive=False,
                        code=False,
                        umd=False,
                        quantitative=False,
                    )
                
                    updated_polygon = dataclasses.replace(poly_data, semantic_clasification=semantic_obj)
                    self.workflow.polygons[poly_id] = updated_polygon
                    created_count += 1
                    logger.debug(f"Clasificación fallback creada para {poly_id}: {semantic_obj}")
            
            if created_count > 0:
                logger.debug(f"Creadas {created_count} clasificaciones semánticas fallback")
            else:
                logger.debug("Todos los polígonos ya tenían clasificación semántica")
            
            return True
                
        except Exception as e:
            logger.error(f"Error creando semantic_clasification: {e}", exc_info=True)
            return False
        
    def update_semantic_clasification(self, final_results: Dict[str, SemanticClassification], reset_refined: bool = False) -> bool:
        """
        Actualiza el semantic_clasification de los polígonos.
        
        Args:
            final_results: Diccionario {poly_id: SemanticClassification}
            reset_refined: Si True, resetea was_refined=False después de actualizar
        """
        try:
            if not self.workflow:
                logger.error("No hay workflow inicializado para actualizar resultados OCR.")
                return False

            updated_count = 0

            for poly_id, semantic_object in final_results.items():
                if poly_id in self.workflow.polygons:
                    polygon = self.workflow.polygons[poly_id]

                    # Actualizar semantic_clasification y opcionalmente resetear was_refined
                    if reset_refined:
                        updated_polygon = dataclasses.replace(
                            polygon, 
                            semantic_clasification=semantic_object,
                            was_refined=False
                        )
                    else:
                        updated_polygon = dataclasses.replace(polygon, semantic_clasification=semantic_object)
                    
                    self.workflow.polygons[poly_id] = updated_polygon
                    updated_count += 1

            if updated_count > 0:
                logger.debug(f"Actualizados {updated_count} polígonos con semantic_clasifications (reset_refined={reset_refined})")
            return True
            
        except Exception as e:
            logger.error(f"Error actualizando múltiples polígonos: {e}", exc_info=True)
            return False
        
    def update_key_field(self, polygon_updates: Optional[Dict[str, str]]) -> bool:
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
            
                    logger.debug(f"UPDATED: poly_id={poly_id}, key_field={key_field}, text='{polygon.ocr_text or ''}'")

            if updated_count > 0:
                logger.debug(f"Actualizados {updated_count} polígonos con key_fields")
                return True
            
            else:
                logger.warning("No hubo poligonos con key_field")
                return False
            
        except Exception as e:
            logger.warning(f"Error actualizando múltiples polígonos: {e}", exc_info=True)
            return False

    def _find_and_mark_header(self) -> Optional[str]:
        """Localiza la line_id del encabezado basada en HeaderWords y marca header_line=True."""
        try:
            if not self.workflow:
                return None
            
            polygons = self.workflow.polygons if self.workflow else{}
            all_lines = self.workflow.all_lines if self.workflow else{}

            hdr_poly_ids: List[str] = [pid for pid, p in polygons.items() if getattr(p, "key_field", None) == "HeaderWords"] 

            logger.debug(f"Header_polys: {hdr_poly_ids}")
            
            # LOG CRÍTICO: Verificar si all_lines tiene datos
            if not all_lines:
                logger.error("all_lines está vacío! No se puede buscar el header.")
                return None
            
            # Buscar la línea que contiene el mayor número de polígonos de encabezado
            header_line_id = None
            max_header_count = 0 

            for line_id, line_data in all_lines.items():
                polygon_ids = getattr(line_data, "polygon_ids", [])
                # Contar cuántos polígonos de encabezado están en esta línea
                header_count = sum(1 for pid in hdr_poly_ids if pid in polygon_ids)
                
                # LOG: Mostrar líneas que tienen al menos un HeaderWord
                if header_count > 0:
                    logger.debug(f"Línea {line_id} tiene {header_count} HeaderWords: {[pid for pid in hdr_poly_ids if pid in polygon_ids]}")

                if header_count > max_header_count:
                    # Si esta línea tiene más polígonos de encabezado, actualizar
                    header_line_id = line_id
                    max_header_count = header_count

            if header_line_id is not None:
                # Marcar la línea como header
                current = self.workflow.all_lines.get(header_line_id)
                if current:
                    updated_line = dataclasses.replace(current, header_line=header_line_id)
                    self.workflow.all_lines[header_line_id] = updated_line
                    
                    # Actualizar todos los polígonos según la línea de encabezado
                    self.update_headers(header_line_id)
                    
                    logger.debug(f"Header_line_id={header_line_id} guardado correctamente")
                    return header_line_id
            else:
                logger.warning(f"No se encontró ninguna línea con HeaderWords. hdr_poly_ids={hdr_poly_ids}")
        
        except Exception as e:
            logger.error(f"No hubo encabezado textual por similitud de encabezado: {e}", exc_info=True)
            return None

    def update_headers(self, header_line_id: str) -> bool:
        """
        Actualiza los key_field de todos los polígonos según la línea de encabezado:
        1. Todos los polígonos de la línea de encabezado → key_field="HeaderWords"
        2. Todos los demás polígonos con key_field="HeaderWords" → key_field=None
        """
        try:
            if not self.workflow:
                return False
            
            # Obtener los polygon_ids de la línea de encabezado
            header_line = self.workflow.all_lines.get(header_line_id)
            if not header_line:
                logger.warning(f"No se encontró la línea de encabezado {header_line_id}")
                return False
            
            header_polygon_ids = set(getattr(header_line, "polygon_ids", []))
            
            marked_as_header = 0
            cleared_header = 0
            
            # Actualizar todos los polígonos
            for poly_id, polygon in self.workflow.polygons.items():
                current_key_field = getattr(polygon, "key_field", None)
                
                if poly_id in header_polygon_ids:
                    # Este polígono pertenece a la línea de encabezado
                    if current_key_field != "HeaderWords":
                        updated_polygon = dataclasses.replace(polygon, key_field="HeaderWords")
                        self.workflow.polygons[poly_id] = updated_polygon
                        marked_as_header += 1
                        logger.debug(f"Polígono {poly_id} marcado como HeaderWords (en línea {header_line_id})")
                else:
                    # Este polígono NO pertenece a la línea de encabezado
                    if current_key_field == "HeaderWords":
                        updated_polygon = dataclasses.replace(polygon, key_field=None)
                        self.workflow.polygons[poly_id] = updated_polygon
                        cleared_header += 1
                        logger.debug(f"Polígono {poly_id} limpiado de HeaderWords (fuera de línea {header_line_id})")
            
            logger.debug(f"Actualización de polígonos de encabezado: {marked_as_header} marcados, {cleared_header} limpiados")
            return True
            
        except Exception as e:
            logger.error(f"Error actualizando polígonos de encabezado: {e}", exc_info=True)
            return False
            
    def _get_footer(self) -> Optional[str]:
        try:
            if not self.workflow:
                return None
            
            polygons = self.workflow.polygons if self.workflow else{}
            all_lines = self.workflow.all_lines if self.workflow else{}
            footer_poly_ids: List[str] = [pid for pid, p in polygons.items() if getattr(p, "key_field", None) in ("TotalProductos", "MontoTotalDocumento")]
            header_line_ids = [lid for lid, l in all_lines.items() if getattr(l, "header_line", None) is not None]
            header_line_id = header_line_ids[0] if header_line_ids else None
            
            footer_line = None
            polyid_to_lineid: Dict[str, str] = {}
            for pid in footer_poly_ids:
                for line_id, line_obj in all_lines.items():
                    if pid in line_obj.polygon_ids:
                        polyid_to_lineid[pid] = line_id
                        break
                    
            # Elegir el footer más cercano al header_line_id o de menor valor si es que no hay 
            min_distance = None
            for pid, line_id in polyid_to_lineid.items():
                if line_id in all_lines:
                    idx = list(all_lines.keys()).index(line_id)
                    
                    if header_line_id is not None:
                        header_idx = list(all_lines.keys()).index(header_line_id)
                        distance = abs(idx - header_idx)
                        if min_distance is None or distance < min_distance:
                            min_distance = distance
                            footer_line = line_id
                            
                    if header_line_id is None:
                        min_idx = 1
                        distance = abs(idx - min_idx)    
                        if min_distance is None or distance < min_distance:
                            min_distance = distance
                            footer_line = line_id
                            
            if footer_line is not None:
                current = self.workflow.all_lines.get(footer_line)
                if current:
                    updated_line = dataclasses.replace(current, footer_line=footer_line)
                    self.workflow.all_lines[footer_line] = updated_line
                    
                    return footer_line
            else: 
                logger.warning(f"No se encontró ninguna línea para pie de tabla")
                
            return None
                
        except Exception as e:
            logger.info(f"Error buscando footer: {e}", exc_info=True)
            return None
            
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
                    line_centroid=line_data.get("line_centroid", [0, 0]),
                    line_bbox=line_data.get("line_bbox", [0, 0, 0, 0])
                )
                
                all_lines_dataclasses[line_id] = AllLines(
                    lineal_id=line_id,
                    text=line_data.get("text", ""),
                    polygon_ids=line_data.get("polygon_ids", []),
                    line_geometry=line_geometry,
                    tabular_line=False,
                    header_line=None,
                    footer_line=None,
                )
            
            self.workflow.all_lines = all_lines_dataclasses

            header_line = self._find_and_mark_header()
            footer_line = self._get_footer()
            
            if header_line is None or footer_line is None:
            
                logger.info(F"No se encontró encabezado")
                return True
                
            logger.info(f"Header marcado automáticamente: {header_line}")
            logger.info(f"Footer marcado automáticamente: {footer_line}")
            return True
                        
        except Exception as e:
            logger.error(f"Error guardando líneas de texto: {e}", exc_info=True)
            return False

    def save_tabular_lines(self, line_ids: List[str]) -> bool:
        """
        Identifica las líneas tabulares y las guarda como dataclasses TabularLines
        en el workflow. También actualiza el flag en AllLines.
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
                if line_id in self.workflow.all_lines:
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
                logger.debug(f"Marcadas {marked_count} líneas como tabulares: {marked_ids}")
                for log_debug in tabular_lines_debug:
                    logger.debug(f"{log_debug['line_id']} tabular: '{log_debug['text']}' | polygons: {log_debug['polygon_ids']}")
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

