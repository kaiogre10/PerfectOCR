import numpy as np
from typing import List, Any, Optional, Tuple
from scipy.sparse import csr_matrix # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
import logging

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
    cosine = np.dot(vec_to_other, ref_vec) / (np.linalg.norm(vec_to_other) * np.linalg.norm(ref_vec))
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
        
        cosine_sim = np.dot(vec_to_other, ref_vec) / (np.linalg.norm(vec_to_other) * np.linalg.norm(ref_vec))
        return 1.0 - abs(float(cosine_sim))   

    else:
        return 1.0

def calculate_similarity_ref(X: csr_matrix, ref_vec: np.ndarray[Any, Any], dense_output: bool = False):
    return cosine_similarity(ref_vec, X, dense_output)[0]

def cosine_similarity_global(X: csr_matrix, Y: None=None, dense_output: bool = False):
    return cosine_similarity(X, Y, dense_output).astype(np.float32)

def euclidean_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calcula la distancia euclidiana entre dos puntos en ℝ².
    """
    if point1 != point2:
        return 0.0
    
    return float(np.linalg.norm(np.subtract(point1, point2)))

def calculate_hist(areas_array: np.ndarray[Any, Any]) -> Tuple[np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, np.dtype[np.float32]]]:
    return np.histogram(areas_array, bins=(np.histogram_bin_edges(areas_array, 'fd')).astype(np.float32))

def vectorice_values(data_list: List[float], value: Optional[str]) -> float | List[float]:
    """
    Calcula estadísticas vectorizadas (media, desviación estándar, varianza) de una lista de valores.
    """
    if not data_list:
        if value in ["mean", "std", "var"]:
            return 0.0  # Retorna float para casos específicos
        else:
            return [0.0, 0.0, 0.0]  # Retorna lista [mean, std, var] para caso general

    value_array = np.array(data_list, dtype=np.float32)

    if value == "mean":
        return float(np.mean(value_array))
    
    elif value == "std":
        return float(np.std(value_array))
        
    elif value == "var":
        return float(np.var(value_array))
    
    else:
        line_mean = np.mean(value_array)
        line_std = np.std(value_array)
        line_var = np.var(value_array)
        return [float(line_mean), float(line_std), float(line_var)]
        
def define_intervals(bboxes_array: np.ndarray[Any, Any], overlap_threshold: float) -> List[np.ndarray[Any, Any]]:
    """
    Agrupa bounding boxes en líneas de texto replicando exactamente LinealReconstructor.
    Asume que bboxes_array ya está ordenado por centroide Y (como en lineal_reconstructor.py).
    Usa promedios para Y y min/max para X en el bbox de la línea.
    Ordena cada línea por centroide X.
    Retorna una lista de arrays 2D, cada uno con forma (n, 4) para las bounding boxes de la línea.
    """
    if bboxes_array.shape[0] == 0:
        return []

    line_groups: List[np.ndarray[Any, Any]] = []
    
    # Asumir que bboxes_array ya está ordenado por centroide Y (no reordenar)
    # Usar índices directos (0, 1, 2, ...)
    
    # 3. Inicializar la primera línea
    current_line_indices = [0]  # Índices directos
    current_sum_y1 = float(bboxes_array[0, 1])
    current_sum_y2 = float(bboxes_array[0, 3])
    current_count = 1
    current_min_x = float(bboxes_array[0, 0])
    current_max_x = float(bboxes_array[0, 2])

    # 4. Iterar sobre el resto
    for i in range(1, len(bboxes_array)):
        bbox = bboxes_array[i]
        bbox_y1 = float(bbox[1])
        bbox_y2 = float(bbox[3])
        bbox_x1 = float(bbox[0])
        bbox_x2 = float(bbox[2])
        
        # Calcular límites de la línea usando promedios en Y y min/max en X
        line_y1 = current_sum_y1 / current_count
        line_y2 = current_sum_y2 / current_count
        
        # Calcular solapamiento vertical
        overlap_abs = max(0.0, min(line_y2, bbox_y2) - max(line_y1, bbox_y1))
        min_h = min(line_y2 - line_y1, bbox_y2 - bbox_y1)
        overlap = overlap_abs / min_h if min_h > 1e-5 else 0.0
        
        if overlap > overlap_threshold:
            # Agregar a la línea
            current_line_indices.append(i)
            current_sum_y1 += bbox_y1
            current_sum_y2 += bbox_y2
            current_count += 1
            current_min_x = min(current_min_x, bbox_x1)
            current_max_x = max(current_max_x, bbox_x2)
        else:
            # Finalizar línea anterior
            group_indices = np.array(current_line_indices)
            
            # Obtener las bounding boxes del grupo
            group_bboxes = bboxes_array[group_indices]
            
            # Ordenar por centroide X dentro de la línea
            group_cx = (group_bboxes[:, 0] + group_bboxes[:, 2]) / 2.0
            x_sort_order = np.argsort(group_cx)
            
            # Agregar las bounding boxes ordenadas
            line_groups.append(group_bboxes[x_sort_order])
            
            # Iniciar nueva línea
            current_line_indices = [i]
            current_sum_y1 = bbox_y1
            current_sum_y2 = bbox_y2
            current_count = 1
            current_min_x = bbox_x1
            current_max_x = bbox_x2

    # 5. Guardar la última línea
    if current_line_indices:
        group_indices = np.array(current_line_indices)
        group_bboxes = bboxes_array[group_indices]
        group_cx = (group_bboxes[:, 0] + group_bboxes[:, 2]) / 2.0
        x_sort_order = np.argsort(group_cx)
        line_groups.append(group_bboxes[x_sort_order])

    logger.info(f"Line groups: {np.array(line_groups[0])} lines")
    # logger.info(f"Line groups reshaped:: {np.array(line_groups[0])} lines")
    return line_groups