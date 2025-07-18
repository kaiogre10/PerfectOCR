# PerfectOCR/core/utils/geometric.py
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
import math
from shapely.geometry import Polygon, MultiPolygon # multipolygon y polygon son alias o tipos que podrían usarse
from shapely.ops import unary_union # Para unir polígonos
import logging

logger = logging.getLogger(__name__)

def get_polygon_bounds(polygon_coords: List[List[Union[float, int]]]) -> Tuple[float, float, float, float]:
    """Devuelve (min_x, min_y, max_x, max_y) para un polígono."""
    if not polygon_coords or not isinstance(polygon_coords, list) or \
       not all(isinstance(p, (list, tuple)) and len(p) == 2 for p in polygon_coords):
        logger.debug(f"get_polygon_bounds: Coordenadas inválidas o vacías: {polygon_coords}")
        return 0.0, 0.0, 0.0, 0.0 # Retornar un valor por defecto o lanzar excepción
    try:
        valid_coords = []
        for p in polygon_coords:
            try:
                # Asegurar que las coordenadas internas sean numéricas y convertibles a float
                valid_coords.append([float(p[0]), float(p[1])])
            except (TypeError, ValueError, IndexError) as e_point:
                logger.warning(f"Punto inválido {p} en get_polygon_bounds (coords: {polygon_coords}): {e_point}. Omitiendo punto.")
                continue
        
        if not valid_coords or len(valid_coords) < 1: # Necesita al menos un punto válido
            logger.debug(f"get_polygon_bounds: No hay coordenadas válidas después de la conversión: {polygon_coords}")
            return 0.0,0.0,0.0,0.0

        np_array = np.array(valid_coords, dtype=float)
        min_x = float(np_array[:, 0].min())
        min_y = float(np_array[:, 1].min())
        max_x = float(np_array[:, 0].max())
        max_y = float(np_array[:, 1].max())
        return min_x, min_y, max_x, max_y
    except (IndexError, TypeError, ValueError) as e: # Capturar errores de numpy también
        logger.warning(f"get_polygon_bounds: Error procesando coordenadas {polygon_coords}: {e}")
        return 0.0, 0.0, 0.0, 0.0

def get_polygon_height(polygon_coords: List[List[Union[float, int]]]) -> float:
    """Calcula la altura de un polígono basada en sus bounds."""
    _, min_y, _, max_y = get_polygon_bounds(polygon_coords)
    return max_y - min_y if max_y >= min_y else 0.0

def get_polygon_width(polygon_coords: List[List[Union[float, int]]]) -> float:
    """Calcula el ancho de un polígono basada en sus bounds."""
    min_x, _, max_x, _ = get_polygon_bounds(polygon_coords)
    return max_x - min_x if max_x >= min_x else 0.0

def get_polygon_y_center(polygon_coords: List[List[Union[float, int]]]) -> float:
    """Calcula el centro Y de un polígono."""
    _, min_y, _, max_y = get_polygon_bounds(polygon_coords)
    return (min_y + max_y) / 2.0

def get_polygon_x_center(polygon_coords: List[List[Union[float, int]]]) -> float:
    """Calcula el centro X de un polígono."""
    min_x, _, max_x, _ = get_polygon_bounds(polygon_coords)
    return (min_x + max_x) / 2.0

def get_shapely_polygon(input_geometry: Any) -> Optional[Polygon]:
    """
    Convierte una variedad de entradas geométricas a un objeto Polygon de Shapely validado.
    Acepta una instancia de Polygon, o una lista de coordenadas [[x,y], ...].
    """
    if isinstance(input_geometry, Polygon):
        poly = input_geometry
        if not poly.is_valid:
            logger.debug(f"Polígono Shapely de entrada no era válido, intentando buffer(0). Area original: {poly.area}")
            poly = poly.buffer(0) 
        return poly if poly and not poly.is_empty and poly.is_valid else None

    if not isinstance(input_geometry, list) or len(input_geometry) < 3:
        # logger.debug(f"get_shapely_polygon: Coordenadas de entrada insuficientes o tipo incorrecto: {input_geometry}")
        return None
    try:
        # Asegurar que los puntos sean tuplas de floats para Shapely
        tupled_coords: List[Tuple[float, float]] = []
        for point in input_geometry:
            if isinstance(point, (list, tuple)) and len(point) == 2:
                try:
                    tupled_coords.append((float(point[0]), float(point[1])))
                except (TypeError, ValueError):
                    logger.warning(f"Punto inválido {point} en input_geometry para Shapely, omitiendo.")
                    continue 
            else: 
                logger.warning(f"Formato de punto inesperado {point} en input_geometry para Shapely. Se esperaba lista/tupla de 2 elementos.")
                return None 

        if len(tupled_coords) < 3:
            logger.debug(f"Insuficientes puntos válidos ({len(tupled_coords)}) para crear polígono desde: {input_geometry}")
            return None

        poly = Polygon(tupled_coords)
        if not poly.is_valid:
            logger.debug(f"Polígono Shapely construido no era válido, intentando buffer(0). Area original: {poly.area}")
            poly = poly.buffer(0) 
        
        return poly if poly and not poly.is_empty and poly.is_valid else None
    except Exception as e:
        logger.error(f"Excepción creando/validando polígono de Shapely desde {input_geometry}: {e}", exc_info=False)
        return None

def calculate_iou(poly_coords1: List[List[Union[float, int]]], poly_coords2: List[List[Union[float, int]]]) -> float:
    """Calcula Intersection over Union (IoU) entre dos conjuntos de coordenadas de polígonos."""
    p1 = get_shapely_polygon(poly_coords1)
    p2 = get_shapely_polygon(poly_coords2)

    if not p1 or not p2: 
        return 0.0
    
    try:
        intersection_area = p1.intersection(p2).area
        union_area = p1.area + p2.area - intersection_area
        # Evitar división por cero si union_area es muy pequeña o cero
        return intersection_area / union_area if union_area > 1e-9 else 0.0 
    except Exception as e_topo: 
        logger.warning(f"Error de topología calculando IoU entre {poly_coords1} y {poly_coords2}: {e_topo}", exc_info=False)
        return 0.0

def enrich_word_data_with_geometry(word_data: Dict[str, Any],
                                    page_center_x: float, page_center_y: float) -> Dict[str, Any]:
    """
    Añade/actualiza xmin, ymin, xmax, ymax, cx, cy, height, width, y datos relativos
    al centro de página. Prioriza 'polygon_coords', luego 'bbox'.
    """
    enriched = word_data.copy() # Trabajar sobre una copia
    
    # Determinar la fuente primaria de la geometría
    # Si 'polygon_coords' existe y es válido, usarlo.
    # Si no, intentar con 'bbox' (asumiendo [xmin, ymin, xmax, ymax]).
    
    poly_to_process = None
    if 'polygon_coords' in enriched and isinstance(enriched['polygon_coords'], list) and len(enriched['polygon_coords']) >= 3:
        poly_to_process = enriched['polygon_coords']
    elif 'bbox' in enriched and isinstance(enriched['bbox'], list) and len(enriched['bbox']) == 4:
        # Convertir bbox [xmin, ymin, xmax, ymax] a formato de polígono
        try:
            xmin, ymin, xmax, ymax = [float(c) for c in enriched['bbox']]
            if xmax > xmin and ymax > ymin:
                poly_to_process = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
            else:
                logger.debug(f"Bbox con dimensiones no positivas en palabra: {enriched.get('text','N/A')}, bbox: {enriched['bbox']}")
        except (TypeError, ValueError):
             logger.warning(f"Bbox con coordenadas no numéricas en palabra: {enriched.get('text','N/A')}, bbox: {enriched['bbox']}")
    
    if not poly_to_process:
        logger.debug(f"No se pudo obtener geometría válida (polygon_coords o bbox) para enriquecer palabra: {enriched.get('text','N/A')}")
        return enriched # Devolver sin cambios si no hay geometría base

    try:
        # Usar get_shapely_polygon para validar y obtener un objeto Polygon
        shapely_poly = get_shapely_polygon(poly_to_process)
        if not shapely_poly: # Si get_shapely_polygon devuelve None (inválido, vacío, etc.)
            logger.warning(f"No se pudo crear un polígono Shapely válido para la palabra: {enriched.get('text','N/A')}")
            return enriched

        xmin, ymin, xmax, ymax = shapely_poly.bounds
        centroid = shapely_poly.centroid
        cx, cy = centroid.x, centroid.y
        
        enriched.update({
            'xmin': float(xmin), 'ymin': float(ymin), 'xmax': float(xmax), 'ymax': float(ymax),
            'cx': float(cx), 'cy': float(cy),
            'height': float(max(0.0, ymax - ymin)),
            'width': float(max(0.0, xmax - xmin)),
            'polygon_coords': [list(p) for p in shapely_poly.exterior.coords[:-1]], # Coordenadas del polígono (puede ser el MBR si vino de bbox)
            'rel_cx_page_center': float(cx - page_center_x),
            'rel_cy_page_center': float(cy - page_center_y),
            'dist_to_page_center': float(math.sqrt((cx - page_center_x)**2 + (cy - page_center_y)**2)),
            'angle_to_page_center_rad': float(math.atan2(page_center_y - cy, cx - page_center_x)), # Y invertido para ángulo matemático estándar
            'angle_to_page_center_deg': float(math.degrees(math.atan2(page_center_y - cy, cx - page_center_x)))
        })
    except Exception as e:
        logger.error(f"Error calculando propiedades geométricas para palabra {enriched.get('text','N/A')}: {e}", exc_info=True)
    
    return enriched

def tighten_geometry(input_geometry: Any,
                     shrink_ratio: float = 0.03,
                     min_shrink_px: float = 1.0) -> Optional[Dict[str, Any]]:
    """
    Contrae ligeramente la geometría para que quede más "pegada" al texto.

    Parámetros
    ----------
    input_geometry : list | shapely.geometry.Polygon
        Coordenadas [[x,y], ...] o un Polygon válido.
    shrink_ratio : float
        Fracción de la altura que se restará (0.03 = 3 % por defecto).
    min_shrink_px : float
        Contracción mínima absoluta en píxeles (1 px por defecto).

    Retorna
    -------
    dict con:
        polygon_coords : coords del polígono contraído
        shapely_polygon : objeto Polygon resultante
        bbox : (xmin, ymin, xmax, ymax)
        height_h1 : altura ajustada menos 1 px
    o None si la entrada es inválida.
    """
    poly = get_shapely_polygon(input_geometry)
    if not poly:
        return None

    xmin, ymin, xmax, ymax = poly.bounds
    height = ymax - ymin
    if height <= 1e-5:
        return None

    # ε = % de la altura o mínimo absoluto
    epsilon = max(height * shrink_ratio, min_shrink_px)
    epsilon = min(epsilon, height / 2.0 - 1e-3)  # evita colapsar la figura

    buffered_poly = poly.buffer(-epsilon) if epsilon > 0 else poly
    if buffered_poly.is_empty or not buffered_poly.is_valid:
        buffered_poly = poly  # fallback

    b_min_x, b_min_y, b_max_x, b_max_y = buffered_poly.bounds
    polygon_coords = [list(p) for p in buffered_poly.exterior.coords[:-1]]
    height_h1 = max((b_max_y - b_min_y) - 1.0, 0.0)

    return {
        "polygon_coords": polygon_coords,
        "shapely_polygon": buffered_poly,
        "bbox": (b_min_x, b_min_y, b_max_x, b_max_y),
        "height_h1": height_h1
    }

