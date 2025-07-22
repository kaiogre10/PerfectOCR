# PerfectOCR/core/lineal_finder/line_reconstructor.py
import logging
import math
import numpy as np
from typing import Dict, Any, List, Optional, Union
from shapely.geometry import Polygon
from shapely.ops import unary_union
from core.workspace.utils.geometric import get_shapely_polygon

logger = logging.getLogger(__name__)

class LineReconstructor:
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.corrections = config
                
    def _calculate_geometry(self, polygons: Dict) -> tuple[float, float]:
        """
        Calcula el centroide de un polígono.
        El polígono debe tener el formato:
        - Lista de puntos: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        - O rectángulo contenedor: [xmin, ymin, xmax, ymax]
        """
        box = ['polygons']
        if not box:
            return 0.0, 0.0
            
        if isinstance(box[0], list):  # Formato de lista de puntos
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            cx = sum(xs) / len(xs) if xs else 0.0
            cy = sum(ys) / len(ys) if ys else 0.0
        else:  # Formato de rectángulo contenedor [xmin, ymin, xmax, ymax]
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            
        
        y1_min = float(box.get('ymin', 0.0))
        y1_max = float(box.get('ymax', 0.0))
        y2_min = float(box.get('ymin', 0.0))
        y2_max = float(box.get('ymax', 0.0))

        h1 = y1_max - y1_min
        h2 = y2_max - y2_min
        
        
        return cx, cy, h1, h2

    def _group_polygons(self, polygon: List['polygons']) -> Dict[str, Any]:
        """
        Agrupa los polígonos usando bandas Y adaptativas.
        """
        if not polygon:
            return {"grouped_lines": []}

        # Calcula centroides para todos los polígonos
        for p in polygon:
            p['centroide'] = self._calculate_geometry(p)

        # Ordena por centroide Y (de arriba a abajo)
        sorted_polygons = sorted(polygon, key=lambda p: (p['centroide'][1], p['centroide'][0]))
        lineas = []
        linea_actual = []
        if not sorted_polygons:
            return {"grouped_lines": []}

        # Inicializa con el primer polígono
        p0 = sorted_polygons[0]
        y_min = min([y for x, y in p0['polygons']])
        y_max = max([y for x, y in p0['polygons']])
        linea_actual.append(p0)
        
        if h1 <= 1e-5 or h2 <= 1e-5: return False
        overlap_abs = max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))
        if overlap_abs <= 1e-5: return False
            
        min_h = min(h1, h2)
        if min_h <= 1e-5 : return False
        overlap_rel = overlap_abs / min_h

        for p in sorted_polygons[1:]:
            cy = p['centroide'][1]
                # Actualiza el intervalo de la línea
            y_min = min(y_min, p_y_min)
            y_max = max(y_max, p_y_max)
        else:
            lineas.append(linea_actual)
            linea_actual = [p]
            y_min = p_y_min
            y_max = p_y_max
        if linea_actual:
            lineas.append(linea_actual)

        # Output clásico (solo las words agrupadas)
        lineas_finales = []
        for linea in lineas:
            linea_ordenada = sorted(linea, key=lambda p: p['centroide'][0])
            for p in linea_ordenada:
                del p['centroide']
            lineas_finales.append(linea_ordenada)

        return {
            "grouped_lines": lineas_finales
        }
    

    def _calculate_vertical_overlap_ratio(self, el1: Dict, el2: Dict) -> bool:
        

    def _group_elements_by_vertical_overlap(self, elements: List[Dict]) -> List[List[Dict]]:
        if not elements: return []
        elements_sorted = sorted(elements, key=lambda el: float(el.get('ymin', float('inf'))))
        
        groups: List[List[Dict]] = []
        if not elements_sorted: return groups

        current_group: List[Dict] = [elements_sorted[0]]
        for i in range(1, len(elements_sorted)):
            word_to_check = elements_sorted[i]
            can_be_grouped = any(self._calculate_vertical_overlap_ratio(word_to_check, member) for member in current_group)
            
            if can_be_grouped:
                current_group.append(word_to_check)
            else:
                groups.append(current_group)
                current_group = [word_to_check]
        if current_group: groups.append(current_group)
        
        return groups

    def _prepare_ocr_element(self, ocr_item: Dict, element_idx: int, engine_name: str, item_type: str) -> Optional[Dict]:
        try:

            polygon_coords_raw: Optional[List[List[Union[int, float]]]] = None

            if engine_name.lower() == "paddle":
                polygon_coords_raw = ocr_item.get("polygon_coords")
                if not (isinstance(polygon_coords_raw, list) and len(polygon_coords_raw) >= 3):
                    return None
            
            if not polygon_coords_raw: 
                logger.warning(f"polygon_coords_raw es None o vacío: item {element_idx}').")
                return None

            float_poly_coords: List[List[float]] = []
            
            try:
                if not all(isinstance(p, (list, tuple)) and len(p) == 2 for p in polygon_coords_raw):
                    logger.warning(f"Formato de punto inesperado en polygon_coords_raw {element_idx}. Coords: {polygon_coords_raw}")
                    return None
                float_poly_coords = [[float(p[0]), float(p[1])] for p in polygon_coords_raw]
            except (TypeError, ValueError, IndexError) as e:
                logger.warning(f"No se pudieron convertir polygon_coords a float para {engine_name} item {element_idx} Coords: {polygon_coords_raw}. Error: {e}")
                return None
            
            if len(float_poly_coords) < 3:
                logger.warning(f"Pocos puntos válidos ({len(float_poly_coords)}) después de item {element_idx}).")
                return None

            shapely_poly = get_shapely_polygon(float_poly_coords)
            if not shapely_poly or shapely_poly.is_empty or not shapely_poly.is_valid:
                logger.warning(f"Shapely polygon inválido/vacío: item {element_idx}). Coords: {float_poly_coords}")
                return None

            min_x, min_y, max_x, max_y = shapely_poly.bounds
            height = max_y - min_y
            width = max_x - min_x

            if height <= 1e-5 or width <= 1e-5:
                logger.debug(f"Elemento ID {element_idx} con altura/anchura no positiva: h={height:.2f}, w={width:.2f}")
                return None

            centroid = shapely_poly.centroid
            
            return {
                "internal_id": f"{element_idx:04d}",
                "shapely_polygon": shapely_poly,
                "polygon_coords": float_poly_coords,
                "cx": float(centroid.x), "cy": float(centroid.y),
                "xmin": float(min_x), "ymin": float(min_y), 
                "xmax": float(max_x), "ymax": float(max_y),
                "height": float(height), "width": float(width)
            }
        except Exception as e:
            logger.error(f"Error crítico preparando elemento {engine_name} ID {element_idx} ('{ocr_item.get('text','N/A')}'): {e}", exc_info=True)
            return None

    def _build_line_output(self, group: List[Dict], line_idx: int, engine_name: str) -> Optional[Dict[str, Any]]:
        if not group: return None
        group_sorted = sorted(group, key=lambda el: float(el.get('xmin', self.page_width + 1)))
        text = " ".join([el.get('text_raw', '') for el in group_sorted]).strip()
        if not text: return None

        bbox_coords: List[List[float]] = []
        line_geom_props: Dict[str, Optional[float]] = {
            "cy_avg": None, "ymin_line": None, "ymax_line": None, "xmin_line": None, 
            "xmax_line": None, "height_line": None, "width_line": None
        }

        valid_polys = [el.get('shapely_polygon') for el in group_sorted if el.get('shapely_polygon') and el.get('shapely_polygon').is_valid]
        if valid_polys:
            try:
                merged_geom = unary_union(valid_polys)
                if not merged_geom.is_empty and merged_geom.is_valid:
                    final_poly = merged_geom.convex_hull if hasattr(merged_geom, 'convex_hull') and merged_geom.convex_hull.is_valid and not merged_geom.convex_hull.is_empty else merged_geom
                    if final_poly and not final_poly.is_empty and final_poly.is_valid:
                        bbox_coords = [list(coord) for coord in final_poly.exterior.coords[:-1]]
                        b_min_x, b_min_y, b_max_x, b_max_y = final_poly.bounds
                        centroid = final_poly.centroid # Mantener esta línea
                        line_geom_props.update({
                            "cx_avg": float(centroid.x) if centroid else (float(b_min_x) + float(b_max_x)) / 2.0, # AÑADIDO/MODIFICADO
                            "cy_avg": float(centroid.y) if centroid else (float(b_min_y) + float(b_max_y)) / 2.0,
                            "ymin_line": float(b_min_y), "ymax_line": float(b_max_y),
                            "xmin_line": float(b_min_x), "xmax_line": float(b_max_x),
                            "height_line": float(b_max_y - b_min_y), "width_line": float(b_max_x - b_min_x)
                        })

            except Exception as e_union:
                logger.warning(f"Excepción en unary_union para línea {engine_name} idx {line_idx} ('{text[:30]}...'): {e_union}. Usando fallback.")
                bbox_coords = [] 

        if not bbox_coords: 
            xmins = [float(el['xmin']) for el in group_sorted if el.get('xmin') is not None]
            ymins = [float(el['ymin']) for el in group_sorted if el.get('ymin') is not None]
            xmaxs = [float(el['xmax']) for el in group_sorted if el.get('xmax') is not None]
            ymaxs = [float(el['ymax']) for el in group_sorted if el.get('ymax') is not None]

            if xmins and ymins and xmaxs and ymaxs:
                min_x_val, max_x_val = min(xmins), max(xmaxs)
                min_y_val, max_y_val = min(ymins), max(ymaxs)
                if max_x_val > min_x_val and max_y_val > min_y_val:
                    bbox_coords = [
                        [min_x_val, min_y_val], [max_x_val, min_y_val],
                        [max_x_val, max_y_val], [min_x_val, max_y_val]
                    ]
                    cxs_fb = [float(el.get('cx',0.0)) for el in group_sorted if el.get('cx') is not None] # AÑADIDO
                    cys_fb = [float(el.get('cy',0.0)) for el in group_sorted if el.get('cy') is not None]
                    line_geom_props.update({
                        "cx_avg": np.mean(cxs_fb) if cxs_fb else (min_x_val + max_x_val) / 2.0, # AÑADIDO
                        "cy_avg": np.mean(cys_fb) if cys_fb else (min_y_val + max_y_val) / 2.0,
                        "ymin_line": min_y_val, "ymax_line": max_y_val,
                        "xmin_line": min_x_val, "xmax_line": max_x_val,
                        "height_line": max_y_val - min_y_val, "width_line": max_x_val - min_x_val
                    })
                    
                else: 
                    logger.warning(f"Fallback de Bbox degenerado para línea idx {line_idx}")
                    bbox_coords = [] 
            else:
                logger.warning(f"No se pudo determinar geometría (fallback) para línea idx {line_idx}")
                bbox_coords = []
        
        if not bbox_coords:
            logger.error(f"No se pudo determinar la geometría para la línea idx {line_idx}")
            return None
        
        output_elements = []
        for el in group_sorted:
            clean_el = el.copy()
            clean_el.pop('shapely_polygon', None)
            output_elements.append(clean_el)

        logger.debug(f"[LineReconstructor] Línea {line_idx}")

        return {
            'line_id': f"line_{line_idx:04d}", 
            'polygon_line_bbox': bbox_coords,
            'geometric_properties_line': line_geom_props, 
            'reconstruction_source': f"overlap_grouping"
        }

    def _reconstruct_lines(self, polygons: List[List[List[float]]], metadata: tuple[int, int], int]) -> List[List[List[List[float]]]]:
        logger.info(f"Iniciando reconstrucción de líneas")
        prepared_elements = [el for el in [self._prepare_ocr_element(item, i, item_type) 
                                           for i, item in enumerate(raw_ocr_elements)] if el]
        
        if not prepared_elements:
            logger.warning(f"No se prepararon polígonos.")
            return []

        groups = self._group_elements_by_vertical_overlap(prepared_elements)
        
        reconstructed_lines: List[Dict[str, Any]] = []
        for i, group in enumerate(groups):
            line_obj = self._build_line_output(group, i)
            if line_obj:
                reconstructed_lines.append(line_obj)

        return reconstructed_lines
