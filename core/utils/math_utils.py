import numpy as np
from typing import List, Any, Optional, Tuple, Dict, Sequence
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
import logging
from sklearn.cluster import DBSCAN # type: ignore
from core.utils.data_utils import DENSITY_ENCODER, CHAR_FRECUENCY, CHAR_NUM

logger = logging.getLogger(__name__)

char_num = CHAR_NUM
density = DENSITY_ENCODER
frecuency = CHAR_FRECUENCY

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
    ref_vec = np.array([1, 0]).astype(np.float32)  # eje X positivo
    if np.linalg.norm(vec_to_other) == 0.0:
        return 1.0
    cosine = np.dot(vec_to_other, ref_vec) / (np.linalg.norm(vec_to_other) * np.linalg.norm(ref_vec), np.float32)
    return 1.0 - abs(float(cosine))

def encode_text(text: str, encoder: Dict[str, float]) -> List[float]:
    try:
        if not text:
            return []

        compact_text = ''.join(text.split())
        encoded_poly = [encoder.get(char, 0) for char in compact_text]

        return encoded_poly

    except Exception as e:
        logger.warning(f"Error codificando polígonos: {e}", exc_info=True)
    return []
                
def get_morphological_encode(text: str) -> List[float]:
    try:
        result: List[float] = []
        for ch in text:
            if ch in char_num:
                result.append(1.0)
            elif ch.isalpha():
                result.append(-1.0)
            else:
                result.append(0.0)
        return result

    except Exception as e:
        logger.warning(f"Error codificando polígonos: {e}", exc_info=True)
    return []
    
def text_encode(text: str, encoding_type: List[str]) -> np.ndarray[Any, np.dtype[np.float32]]:
    if "all" in encoding_type and len(encoding_type) == 1:
        encoding_type = ["density", "morphological", "frequency"]

    encoders: List[List[float]]= []
    for enc_type in encoding_type:

        if enc_type == "density":
            dense = encode_text(text.lower(), density)
            encoders.append(dense)
        elif enc_type == "frequency":
            frec = encode_text(text, frecuency)
            encoders.append(frec)
        elif enc_type == "morphological":
           morph = get_morphological_encode(text)
           encoders.append(morph)
    
    return np.array(encoders, np.float32)

def get_cosine_similarity(X: np.ndarray[Any, np.dtype[np.float32]], ref_vec: Optional[np.ndarray[Any, np.dtype[np.float32]]] = None, dense_output: bool = False) -> np.ndarray[Any, np.dtype[np.float32]]:
    """
    Calcula la matriz de similitudes coseno entre los vectores de X y ref_vec. 
    Por convención, X es la matriz de consulta (shape n_samples_X, n_features), 
    ref_vec es la de referencia (shape n_samples_ref, n_features) o None (usa X contra sí misma).
    El resultado será (n_samples_X, n_samples_ref).
    """
    return cosine_similarity(X, ref_vec, dense_output=dense_output) # type: ignore

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
    # time_h = time.perf_counter()
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
    labels: np.ndarray[Any, Any] = clustering.fit_predict(features) #type: ignore
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
