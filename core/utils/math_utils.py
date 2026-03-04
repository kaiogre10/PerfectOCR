import numpy as np
from typing import List, Any, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
import logging
import time
from sklearn.cluster import DBSCAN # type: ignore

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
    vec_to_other = np.array([other_c[0] - ref_point[0], other_c[1] - ref_point[1]]).astype(np.float32)
    ref_vec = np.array([1, 0]).astype(np.float32)  # eje X positivo
    if np.linalg.norm(vec_to_other) == 0.0:
        return 1.0
    cosine = np.dot(vec_to_other, ref_vec) / (np.linalg.norm(vec_to_other) * np.linalg.norm(ref_vec)).astype(np.float32)
    return 1.0 - abs(float(cosine))
    
def bbox_alignment(current_coord: float, other_bbox: List[float], coord_idx: int) -> Optional[float]:
    """
    Mide alineación usando similitud coseno.
    Punto de referencia: [current_coord, 0] en el eje X
    Vector hacia otra línea: [other_coord - current_coord, other_y - 0]
    """
    if other_bbox:
        # Punto de referencia en el eje X
        ref_point = np.array([current_coord, 0.0])

        # Coordenada de la otra línea
        other_coord = other_bbox[coord_idx]  # Acceso correcto por índice
        other_y = other_bbox[1]  # Coordenada Y de la otra línea

        # Vector desde el punto de referencia hacia la otra línea
        vec_to_other = np.array([other_coord - current_coord, other_y - ref_point[1]])
        
        # Vector de referencia (eje X positivo)
        ref_vec = np.array([1, 0])
        
        # Similitud coseno
        if np.linalg.norm(vec_to_other) == 0:
            return 1.0
        
        cosine_sim = np.dot(vec_to_other, ref_vec) / (np.linalg.norm(vec_to_other) * np.linalg.norm(ref_vec)).astype(np.float32)
        return 1.0 - abs(float(cosine_sim))   

    else:
        return 1.0

def get_cosine_similarity(ref_vec: Optional[np.ndarray[Any, np.dtype[np.float32]]] = None, X: np.ndarray[Any, np.dtype[np.float32]] = np.ndarray[Any, np.dtype[np.float32]], dense_output: bool = False) -> np.ndarray[Any, np.dtype[np.float32]]:
    return cosine_similarity(X, ref_vec, dense_output).astype(np.float32)

def euclidean_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calcula la distancia euclidiana entre dos puntos en ℝ².
    """
    if point1 != point2:
        return 0.0
    
    return float(np.linalg.norm(np.subtract(point1, point2)))
        
def extract_contours_histogram(metrics: np.ndarray[Any, Any]) -> Tuple[int, float]:
    """
    Calcula histograma de áreas de contornos de forma recursiva.
    Elimina outliers hasta que no queden más gaps en el histograma.
    Retorna:
        deleted: Número total de outliers eliminados
    """
    time_h = time.perf_counter()
    min_feat = np.min(metrics) if np.min(metrics) == 0 else (np.min(metrics) - 0.1)
    
    def recursive_cleanup(current_metrics: np.ndarray[Any, Any], total_deleted: int, iteration: int) -> Tuple[int, int]:
        """
        Función recursiva que elimina outliers iterativamente.
        """
        current_count = current_metrics.shape[0]
        max_feat = (np.max(current_metrics) + 0.1)
        
        hist, bin_edges = np.histogram(current_metrics, bins=(np.histogram_bin_edges(current_metrics, 'fd', (min_feat, max_feat))).astype(np.float32))
        relat = np.sum(hist)/np.max(hist)
        # logger.info(f"Relación {relat}")
        
        # logger.info(f"HIST iteración {iteration}: {hist}, elementos: {current_count}")

        hist_rever = hist[::-1].astype(np.int32)
        cutting = np.where(hist_rever > 2)[0]
        idx_orig = len(hist) - 1 - cutting[0] if cutting.size > 0 else -1
        outliers_indx = np.nonzero((hist == 1) | (hist == 2))[0]
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
        cond = current_metrics < ind_big
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
    labels: np.ndarray[Any, Any] = clustering.fit_predict(features)
    return labels
    
def dilate_contour(contour: np.ndarray[Any, np.dtype[np.int32]], kernel: np.ndarray[Any, Any])-> np.ndarray[Any, np.dtype[np.int32]]:
    """
    Expande un contorno cerrado según kernel anisótropo.
    
    Args:
        contour: np.ndarray shape (N, 2) dtype int32, puntos del contorno
        kernal: expansión horizontal y vertical en píxeles (kernel cols)
    Returns:
        np.ndarray shape (N, 2) dtype int32, contorno expandido
    """
    pts = contour.astype(np.float32)
    n = len(pts)
    
    if n < 3:
        return contour
    
    next_idx = np.arange(1, n + 1) % n
    prev_idx = np.arange(-1, n - 1) % n
    
    e1 = pts - pts[prev_idx]
    e2 = pts[next_idx] - pts
    
    signed_area = 0.5 * np.sum(pts[:, 0] * pts[next_idx, 1] - pts[next_idx, 0] * pts[:, 1])
    sign = 1.0 if signed_area < 0 else -1.0
    
    def normal_ext(e: np.ndarray) -> np.ndarray:
        length = np.linalg.norm(e, axis=1, keepdims=True) + 1e-9
        unit = e / length
        return sign * np.column_stack([unit[:, 1], -unit[:, 0]])
    
    n1 = normal_ext(e1)
    n2 = normal_ext(e2)
    
    bisect = n1 + n2
    bisect_len = np.linalg.norm(bisect, axis=1, keepdims=True) + 1e-9
    bisect_unit = bisect / bisect_len
    
    cos_half = np.clip(bisect_len.flatten() / 2.0, 0.15, 1.0)
    factor = 1.0 / cos_half
    
    # Anisotropía: escala X e Y por separado
    offset = bisect_unit * factor[:, None] * np.array([[kernel[0], kernel[1]]])
    
    expanded = pts + offset
    return np.round(expanded).astype(np.int32)