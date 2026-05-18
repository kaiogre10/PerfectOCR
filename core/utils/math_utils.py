# PerfectOCR/core/utils/math_utils.py
import numpy as np
import logging
# import pandas as pd
import time
from typing import List, Any, Optional, Tuple, Dict, Sequence, Set
from sklearn.metrics.pairwise import cosine_similarity  # type:ignore
from sklearn.cluster import DBSCAN # type: ignore
from core.utils.data_utils import DENSITY_ENCODER, CUANT_CHAR

density_encoder = DENSITY_ENCODER
cuant_char = CUANT_CHAR

logger = logging.getLogger(__name__)

def alignment(ref_c: List[float], other_c: List[float]) -> float:
    """
    Primer parametro es el valor de referencia que es ortogonal al eje X, mientra que el segundo parametro es que queremos comparar
    Mide alineación usando similitud coseno, tomando como referencia (X,0) del centroide.
    Vector desde (ref_c[0], 0) hacia (other_c[0], other_c[1]).
    """
    if not other_c:
        return 1.0
    ref_point = np.array([ref_c[0], 0.0])
    vec_to_other = np.array([other_c[0] - ref_point[0], other_c[1] - ref_point[1]], np.float32)
    ref_vec = np.array([1, 0], np.float32)  # eje X positivo
    if np.linalg.norm(vec_to_other) == 0.0:
        return 1.0
    cosine = np.dot(vec_to_other, ref_vec) / (np.linalg.norm(vec_to_other) * np.linalg.norm(ref_vec), np.float32)
    return 1.0 - abs(float(cosine))

def get_morphological_encode(text: str) -> np.ndarray[Any, np.dtype[np.float32]]:
    return np.array(list(map(lambda ch: 1.0 if ch in cuant_char else -1.0 if ch.isalpha() else 0.0, text)), np.float32)

def encode_text(text: str, encoder: Dict[str, float]) -> np.ndarray[Any, np.dtype[np.float32]]:
    return np.array([encoder.get(char, " ") for char in text], np.float32)
    
def text_encode(text: str) -> np.ndarray[Any, np.dtype[np.float32]]:
    dense = encode_text(text, density_encoder)
    morph = get_morphological_encode(text)
    # frec = encode_text(text, REL_FRECUENCY_CHAR)
    encoders = np.column_stack([dense, morph])
    return np.mean(encoders, axis=0)

def get_cosine_similarity(X: np.ndarray[Any, np.dtype[np.float32]], ref_vec: Optional[np.ndarray[Any, np.dtype[np.float32]]] = None, dense_output: bool = False) -> np.ndarray[Any, np.dtype[np.float32]]:
    """
    Calcula la matriz de similitudes coseno entre los vectores de X y ref_vec.
    """
    return cosine_similarity(X, ref_vec, dense_output=dense_output) # type: ignore
    
def cosine_similarity_matrix(x: np.ndarray[Any, np.dtype[np.float32]]) -> np.ndarray[Any, np.dtype[np.float32]]:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    X_norm = np.divide(x, norms, out=np.zeros_like(x, dtype=np.float32), where=norms != 0, dtype=np.float32)
    return np.matmul(X_norm, X_norm.T, dtype=np.float32)
    
def mean_cosine_per_row(s: np.ndarray[Any, np.dtype[np.float32]]) -> np.ndarray[Any, np.dtype[np.float32]]:
    return (np.sum(s, axis=1, dtype=np.float32) - 1.0) / (s.shape[0] - 1)

def euclidean_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calcula la distancia euclidiana entre dos puntos en ℝ².
    """
    if point1 != point2:
        return 0.0
    
    return float(np.linalg.norm(np.subtract(point1, point2)))
        
def soft_histogram(metrics: np.ndarray[Any, Any]) -> Tuple[int, float]:
    """
    Suaviza histograma.
    Elimina outliers hasta que no queden más gaps en el histograma.
    deleted: Número total de outliers eliminados
    """
    # time_h = time.perf_counter()
    min_feat = np.min(metrics) if np.min(metrics) == 0 else (np.min(metrics) - 0.1)
    
    def recursive_cleanup(current_metrics: np.ndarray[Any, Any], total_deleted: int, iteration: int) -> Tuple[int, int]:
        """
        Función recursiva que elimina outliers iterativamente.
        """
        current_count = current_metrics.shape[0]
        max_feat = (np.max(current_metrics) + 0.1)
        
        hist, bin_edges = np.histogram(current_metrics, bins=(np.histogram_bin_edges(current_metrics, 'fd', (min_feat, max_feat))))
        relat = np.sum(hist)/np.max(hist)
        # logger.info(f"Relación {relat}")
        # logger.info(f"EDGES: {bin_edges}")
        
        # logger.info(f"HIST iteración {iteration}: {hist}, elementos: {current_count}")

        hist_rever = hist[::-1].astype(np.int32)
        cutting = np.where(hist_rever > 1)[0]
        idx_orig = len(hist) - 1 - cutting[0] if cutting.size > 0 else -1
        outliers_indx = np.nonzero(hist == 1)[0]
        filtered_outliers = outliers_indx[outliers_indx > idx_orig]
        
        # Condición de parada: no hay más outliers
        if filtered_outliers.size == 0:
            # logger.info(f"Analisis de histograma completado en {time.perf_counter()-time_h}'s")
            # logger.info(f"Total eliminados: {total_deleted}")
            # logger.info(f"Iteraciones totales de histograma: {iteration}")
            return total_deleted, relat
        
        # Filtrar outliers
        mask = np.min(filtered_outliers) - 1
        ind_big = bin_edges[mask] 
        cond = ind_big >= current_metrics
        filtered_metrics = np.compress(cond, current_metrics, 0)
        
        new_count = filtered_metrics.shape[0]
        deleted_this_iter = current_count - new_count
        total_deleted += deleted_this_iter
        
        # logger.info(f"Eliminados en iteración {iteration}: {deleted_this_iter}, Total acumulado: {total_deleted}")

        # Llamada recursiva con las métricas filtradas
        return recursive_cleanup(filtered_metrics, total_deleted, iteration + 1)
    
    # Inicia la recursión
    return recursive_cleanup(metrics, 0, 1)
   
def density_cluster(features: np.ndarray[Any, np.dtype[np.float32]], eps: float, min_samples: int, metric: str) ->  np.ndarray[Any, Any]:
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels: np.ndarray[Any, Any] = clustering.fit_predict(features) # type: ignore
    return labels
    
def fragment_geometry_horizontal(geometry: Any, num_fragments: int, proportions: Optional[Sequence[float]] = None) -> List[Dict[str, np.ndarray[Any, Any]]]:
    """
    Fragmenta una geometría en segmentos horizontales (eje X) y devuelve nuevas geometrías
    ordenadas de izquierda a derecha.

    - **No** modifica dataclasses ni accede a manager/workflow.
    - Si `proportions` es None, divide uniforme.
    - Si se provee `proportions`, debe tener longitud `num_fragments` y su suma debe ser > 0.
    """
    if num_fragments <= 1:
        bbox = getattr(geometry, "bounding_box", None)
        centroid = getattr(geometry, "centroid", None)
        coords = getattr(geometry, "polygon_coords", None)
        if bbox is None or centroid is None or coords is None:
            return []
        return [{"bounding_box": bbox, "centroid": centroid, "polygon_coords": coords}]

    bbox = geometry.bounding_box
    if bbox is None or len(bbox) != 4:
        return []

    xmin, ymin, xmax, ymax = map(float, bbox)
    width = xmax - xmin
    height = ymax - ymin
    if width <= 0 or height <= 0:
        return []

    if proportions is None:
        props = np.full((num_fragments,), 1.0 / float(num_fragments), dtype=np.float32)
    else:
        if len(proportions) != num_fragments:
            return []
        props = np.asarray(list(proportions), dtype=np.float32)
        total = float(np.sum(props))
        if total <= 0:
            return []
        props = props / total

    geoms: List[Dict[str, np.ndarray[Any, Any]]] = []
    current_x = xmin

    for i in range(num_fragments):
        frag_width = float(props[i]) * width
        new_xmax = xmax if i == (num_fragments - 1) else (current_x + frag_width)

        new_bbox = np.array([current_x, ymin, new_xmax, ymax], dtype=np.float32)
        new_centroid = np.array([(current_x + new_xmax) / 2.0, (ymin + ymax) / 2.0], dtype=np.float32)
        new_coords = np.array(
            [
                [new_bbox[0], new_bbox[1]],
                [new_bbox[2], new_bbox[1]],
                [new_bbox[2], new_bbox[3]],
                [new_bbox[0], new_bbox[3]],
            ],
            dtype=np.float32,
        )
        geoms.append({"polygon_coords": new_coords, "bounding_box": new_bbox, "centroid": new_centroid})

        current_x = new_xmax

    return geoms

def calculate_features(sorted_lines: List[Any], polygons_dict: Dict[str, Any], img_dims: Tuple[int, int]) -> np.ndarray[Any, Any]:
    """
    Calcula features geométricos + alineación tabular por cada línea.
    """    
    t0 = time.perf_counter()
    
    all_features = calculate_math_features(sorted_lines, img_dims)
    # logger.info("Features completas:"
    # "\n"f"{all_features}"
    # "\n"f"SHAPE:{all_features.shape}")
    
    textual_features = calculate_textual_line_features(sorted_lines, polygons_dict)
    # logger.info(f"Features textuales shape: {textual_features.shape}"
    #             "\n"f"{textual_features[:, -2]}")

    all_lines_features = np.column_stack([all_features, textual_features])
    #logger.debug("TODAS LAS FEATURES"
     #       "\n"f"SHAPE:{all_lines_features.shape}"
      #  "\n"f"{np.array2string(all_lines_features, precision=3, suppress_small=True)}")
            
    logger.debug(f"VECTORIZACIÓN COMPLETADA EN: {time.perf_counter() - t0:.7f}s")

    return all_lines_features

def calculate_global_stats(geoline_features: np.ndarray[Any, Any]) -> np.ndarray[Any, np.dtype[np.float32]]:
    maxs = np.max(geoline_features, axis=0, keepdims=True)
    medians = np.median(geoline_features, axis=0, keepdims=True)
    mins = np.min(geoline_features, axis=0, keepdims=True)
    return np.column_stack([maxs, medians, mins])

def calculate_math_features(sorted_lines: List[Any], img_dims: Tuple[int, int])-> np.ndarray[Any, np.dtype[np.float32]]:
    # timeall = time.perf_counter()
    total_width = img_dims[1] or 0.0
    total_height = img_dims[0] or 0.0
    total_size = total_width * total_height
    
    line_id = np.array([lid.line_index for lid in sorted_lines])
    geometry = [lid.line_geometry for lid in sorted_lines]
    all_bboxes = np.array([geo.line_bbox for geo in geometry], np.float32)
    x, y, w, h = all_bboxes[:, 0], all_bboxes[:, 1], all_bboxes[:, 2], all_bboxes[:, 3]
    width = (w - x)
    height = (h - y)
    area = (width * height)
    perimeter = 2.0 * (width + height)
    aspect_ratio = (height / width) * 100.0
    diagonal = np.sqrt((width**2.0) + (height**2.0))
    angle = np.degrees(np.arctan2(h, w))
    
    global_stats = calculate_global_stats(np.column_stack([width, height, area, perimeter, aspect_ratio, diagonal, angle]))

    # Funciones helpers para división segura igualando la lógica de "if x != 0 else 0.0"
    def safe_div(a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]):
        return np.divide(a, b, out=np.zeros_like(a), where=b!=0)
    
    def safe_dif(val: np.ndarray[Any, Any], med: np.ndarray[Any, Any]):
        return np.where(med != 0, 1 - np.abs(val - med) / med, 0.0)

    # Reemplazos con división segura
    bbox_height_inv = safe_div(height, global_stats[:, 8])
    bbox_h_dif = safe_dif(height, global_stats[:, 8])
    
    bbox_width_inv = safe_div(width, global_stats[:, 7])
    bbox_w_dif = safe_dif(width, global_stats[:, 7])
    
    norm_wid = safe_div(width, global_stats[:, 0])
    width_rel = safe_div(width, total_width) #type: ignore
    
    area_norm = safe_div(area, global_stats[:, 2])
    ratio_area = safe_div(area, total_size) #type: ignore
    
    area_inv = safe_div(area, global_stats[:, 9])
    area_dif = safe_dif(area, global_stats[:, 9])
    
    max_ratio = safe_div(global_stats[:, 2], total_size) #type: ignore
    ratio_area_norm = safe_div(ratio_area, max_ratio)
    
    # aspect_ratio = geoline_features[:, 5]
    aspcrat_inv_norm = 1 - safe_div(np.abs(aspect_ratio), global_stats[:, 4]) # Nota: vectorize usa abs(ar/max)
    
    perimeter_norm = safe_div(perimeter, global_stats[:, 3])
    perimeter_inv = safe_div(perimeter, global_stats[:, 10])
    perimeter_dif = safe_dif(perimeter, global_stats[:, 10])
    
    diag_inv = safe_div(diagonal, global_stats[:, 12])
    diag_dif = safe_dif(diagonal, global_stats[:, 12])
    
    angle_inv = safe_div(angle, global_stats[:, 13])
    diag_norm = safe_div(diagonal, global_stats[:, 5])
    
    compact = safe_div((perimeter ** 2), area) / 100.0
    
    cw: float = (total_width / 2.0)  # centro horizontal de la imagen
    ch: float = (total_height / 2.0)  # centro vertical de la imagen
    main_centroid = np.tile([cw, ch], (line_id.shape[0], 1))

    # Coordenadas prev/next mediante slicing con padding NaN
    centroids = np.array([geo.line_centroid for geo in geometry], np.float32)
    prev_bboxes = np.vstack([np.full((1, 4), np.nan), all_bboxes[:-1]])
    next_bboxes = np.vstack([all_bboxes[1:], np.full((1, 4), np.nan)])
    prev_centroids = np.vstack([np.full((1, 2), np.nan), centroids[:-1]])
    next_centroids = np.vstack([centroids[1:], np.full((1, 2), np.nan)])

    # Coordenadas xmin/xmax actuales
    current_xmin = x
    current_xmax = w

    def _compute_bbox_align(curr_coord: np.ndarray[Any, np.dtype[np.float32]], other_bbox: np.ndarray[Any, np.dtype[np.float32]], idx: int) -> np.ndarray[Any, np.dtype[np.float32]]:
        """
        Versión vectorizada que copia matemáticamente bbox_alignment antiguo.
        Ignora curr_coord[y] y usa 0.0, calculando el coseno con el vector (diferencia_x, other_bbox[y]).
        """
        if other_bbox.shape[0] == 0 or np.all(np.isnan(other_bbox[:, 0])):
            return np.ones_like(curr_coord)
        
        # Calculamos diferencias al estilo ref_point = [curr_coord, 0.0]
        dx = other_bbox[:, idx] - curr_coord
        dy = other_bbox[:, 1] - 0.0  # El y de la otra caja menos 0.0 (fiel a alineación original)
        
        # Vector: [diferencia en X, diferencia en Y desde 0]
        vec = np.column_stack([dx, dy])
        
        # Norma del vector (magnitud para dividir en el coseno)
        norms = np.linalg.norm(vec, axis=1)
        
        # El ref_vec es siempre [1.0, 0.0], por lo que el dot product de vec y ref_vec 
        # siempre es igual a vec[:, 0] (el componente X). La norma de [1,0] es 1.
        with np.errstate(divide='ignore', invalid='ignore'):
            cosine = np.where(norms > 0, vec[:, 0] / norms, 0.0)
        
        result = 1.0 - np.abs(cosine)
        
        # Donde other_bbox es NaN (no existe línea), devolver 1.0 como indicaba "if not prev_bbox else 1.0"
        return np.where(np.isnan(other_bbox[:, idx]), 1.0, result)

    # Pasar current_ymin a las funciones de alineación
    prev_xmin_align: np.ndarray[Any, np.dtype[np.float32]] = _compute_bbox_align(current_xmin, prev_bboxes, 0)
    prev_xmax_align: np.ndarray[Any, np.dtype[np.float32]] = _compute_bbox_align(current_xmax, prev_bboxes, 2)
    next_xmin_align: np.ndarray[Any, np.dtype[np.float32]] = _compute_bbox_align(current_xmin, next_bboxes, 0)
    next_xmax_align: np.ndarray[Any, np.dtype[np.float32]] = _compute_bbox_align(current_xmax, next_bboxes, 2)

    def _compute_centroid_align(ref_c: np.ndarray[Any, np.dtype[np.float32]], other_c: np.ndarray[Any, np.dtype[np.float32]]) -> np.ndarray[Any, np.dtype[np.float32]]:
        """
        Versión vectorizada que copia matemáticamente el alignment antiguo.
        Asume el punto de referencia como [ref_x, 0.0].
        """
        if other_c.shape[0] == 0 or np.all(np.isnan(other_c[:, 0])):
            return np.ones(ref_c.shape[0], np.float32)
        
        # Vector al estilo ref_point = [ref_c[0], 0.0]
        dx = other_c[:, 0] - ref_c[:, 0]
        dy = other_c[:, 1] - 0.0
        
        vec = np.column_stack([dx, dy])
        # Norma del vector
        norms = np.linalg.norm(vec, axis=1)
        
        # Al igual que arriba, el dot product contra [1, 0] es igual al componente X.
        with np.errstate(divide='ignore', invalid='ignore'):
            cosine = np.where(norms > 0, vec[:, 0] / norms, 0.0)
        
        result = 1.0 - np.abs(cosine)
        
        # Donde other_c es NaN (no existe línea), devolver 1.0
        return np.where(np.isnan(other_c[:, 0]), 1.0, result)

    # Aplicar corrección a centroides
    align_prev = _compute_centroid_align(centroids, prev_centroids)
    align_next = _compute_centroid_align(centroids, next_centroids)
    center_align = _compute_centroid_align(centroids, main_centroid)

    all_features = np.column_stack([
        line_id,             # [0] Índice de línea
        bbox_height_inv,     # [1] height normalized/inverse median
        bbox_h_dif,          # [2] diferencia de height vs mediana
        bbox_width_inv,      # [3] width normalized/inverse median
        bbox_w_dif,          # [4] diferencia de width vs mediana
        norm_wid,            # [5] ancho normalizado respecto a máximo
        width_rel,           # [6] ancho relativo al total de imagen
        area_norm,           # [7] area normalizada al máximo

        area_inv,            # [9] area normalizada/inversa de la mediana
        area_dif,            # [10] diferencia de area vs mediana
        center_align,        # [11] alineación con el centroide del documento
        ratio_area_norm,     # [12] ratio relativo a máximo ratio

        aspcrat_inv_norm,    # [14] cuán diferente aspect_ratio vs máximo
        perimeter_norm,      # [15] perímetro normalizado al máximo
        perimeter_inv,       # [16] perímetro inversa/mediana
        perimeter_dif,       # [17] diferencia de perímetro vs mediana
        diag_inv,            # [18] diagonal inversa/mediana
        diag_dif,            # [19] diferencia de diagonal vs mediana
        angle_inv,           # [20] ángulo inversa/mediana
        diag_norm,           # [21] diagonal normalizada al máximo
        compact,             # [22] medida de compactación 
        # slope_inv,          # [23] slope inverso/mediana
        # slope_dif,                     # [24] diferencia de slope vs mediana
        prev_xmin_align,     # [25] 
        prev_xmax_align,     # [26] 
        next_xmin_align,     # [27]
        next_xmax_align,     # [28] 
        align_prev,          # [29] 
        align_next           # [30]
    ])
    # logger.info(f"Todas Las features culculadas en: {time.perf_counter() - timeall:.6}")
    return all_features

def calculate_textual_line_features(sorted_lines: List[Any], polygons_dict: Dict[str, Any]) -> np.ndarray[Any, np.dtype[np.float32]]:
    """
    Devuelve features textuales ajustadas a la lógica de vectorize.py (-1.0/1.0 y conteos correctos).
    """
    # timef = time.perf_counter()
    index_to_id_map = {p.poly_index: p.polygon_id for p in polygons_dict.values()}
    features = np.zeros((len(sorted_lines), 3), np.float32)
    
    for i, line_data in enumerate(sorted_lines):
        sc_quant_count = 0
        kf_total = 0
        # Cuenta tokens numéricos por línea
        poly_ids_line = line_data.polygons_index
        for pid_idx in poly_ids_line:
            pid_str = index_to_id_map.get(pid_idx)
            if pid_str and pid_str in polygons_dict:
                poly = polygons_dict[pid_str]
                kf: Optional[List[int]] = poly.key_field
                if kf is None:
                    sc: List[int] = polygons_dict[pid_str].semantic_clasification
                    sc_quant_count += count_quantitative_tokens(sc)
                    kf_total += 0
                else:
                    kf_total += len(kf)
                
        features[i, 0] = sc_quant_count
        features[i, 1] = 0 if kf_total > 0 else line_data.t_cuant
        features[i, 2] = kf_total
    
    if features.shape[0] == 0:
        return np.zeros((len(sorted_lines), 6), dtype=np.float32)
    
    maximus = np.max(features, axis=0)
    # logger.info("\n"f"{np.column_stack([np.arange(len(sorted_lines)), features])}")
    # logger.info("MAX VALUES:\n"f"{maximus}")
    max_sc_quant, max_digit = maximus[0], maximus[1]
    
    sc_quants, dec_chars, _ = features[:, 0], features[:, 1],features[:, 2]
    
    # Evitar división por cero
    means = np.mean([sc_quants, dec_chars], axis=0)
    dec_mean = means[1]
                
    digit_above = np.where(dec_chars > dec_mean, 1.0, -1.0)
    has_digit = np.where(dec_chars > 1.0, 1.0, -1.0)
    
    if np.all(features[:, 2] < 1):
        has_kf = np.zeros((features[:, 2].size), dtype=np.float32)
    else:
        has_kf = np.where(features[:, 2] == 0, 1.0, -1.0)
    
    if max_sc_quant > 0:
        num_count_norm = (2.0 * sc_quants / float(max_sc_quant)) - 1.0
    else:
        num_count_norm = np.full_like(sc_quants, -1.0, dtype=np.float32)
    
    if max_digit > 0:
        digit_char_frec = (2.0 * dec_chars / float(max_digit)) - 1.0
    else:
        digit_char_frec = np.full_like(dec_chars, -1.0, dtype=np.float32)
    has_quant = np.where(sc_quants > 0, 1.0, -1.0)
    
    if max_digit > 0:
        dig_margin = (dec_chars - dec_mean) / (max_digit / 2.0)
        dig_margin = np.clip(dig_margin, -1.0, 1.0)
    else:
        dig_margin = np.zeros_like(dec_chars, np.float32)
    
    # logger.info(f"Features textuales calculadas en: {time.perf_counter() - timef:.6f}'s")
    textual_features = np.column_stack([dig_margin, has_quant, num_count_norm, digit_above, digit_char_frec, has_digit, has_kf])
    # headers = ["dig_margin", "has_quant", "numeric_count_norm", "digit_above", "digit_char_frec", "has_digit", "has_kf"]
    # textual_df =pd.DataFrame(data=textual_features, columns=headers)
    # logger.info("Features textuales:"
    #            "\n"f"{textual_df.to_string(index=True)}")
    return textual_features

def count_quantitative_tokens(semantic_clasification: List[int]) -> int:
    sc = np.asarray(semantic_clasification, dtype=np.int8)
    mask = (sc == 2) | (sc > 3)
    return 0 if 0 in semantic_clasification else np.count_nonzero(mask)