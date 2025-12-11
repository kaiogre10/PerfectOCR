import numpy as np
import cv2
from typing import List, Any, Optional, Tuple
from scipy.sparse import csr_matrix # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore

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
    
def closest_int(value: float, candidates: List[int]) -> int:
    return min(candidates, key=lambda x: abs(x - value))

def contour_eccentricity(contour: np.ndarray[Any, Any]) -> float:
    if len(contour) < 5:
        return 0.0  # No se puede ajustar una elipse
    ellipse = cv2.fitEllipse(contour)
    (center, axes, angle) = ellipse
    a = max(axes) / 2  # semieje mayor
    b = min(axes) / 2  # semieje menor
    if a == 0:
        return 0.0
    ecc = np.sqrt(1 - (b ** 2) / (a ** 2))
    return float(ecc)
        
def define_intervals(bboxes: np.ndarray[Any, Any], overlap_threshold: float) -> List[np.ndarray[Any, Any]]:
    """
    Agrupa bounding boxes en líneas de texto de manera secuencial, procesándolos
    en orden vertical (de arriba hacia abajo).
    """
    if bboxes.shape[0] == 0:
        return []

    # 1. Obtener los índices originales y ordenar los bboxes por su coordenada Y inicial.
    # Esto asegura que procesamos de arriba hacia abajo.
    original_indices = np.arange(bboxes.shape[0])
    sorted_order = np.argsort(bboxes[:, 1])
    
    sorted_bboxes = bboxes[sorted_order]
    sorted_original_indices = original_indices[sorted_order]

    line_groups: List[np.ndarray[Any, Any]] = []
    if sorted_bboxes.shape[0] == 0:
        return line_groups

    # 2. Inicializar la primera línea con el primer bbox.
    current_line_indices = [sorted_original_indices[0]]
    current_line_bbox = sorted_bboxes[0].copy()

    # Función auxiliar para calcular el solapamiento vertical
    def get_vertical_overlap(bbox1: np.ndarray[Any, Any], bbox2: np.ndarray[Any, Any]) -> float:
        y1_max = max(bbox1[1], bbox2[1])
        y2_min = min(bbox1[3], bbox2[3])
        overlap_height = max(0, y2_min - y1_max)
        
        min_height = min(bbox1[3] - bbox1[1], bbox2[3] - bbox2[1])
        if min_height <= 1e-6: # Evitar división por cero
            return 0.0
        return overlap_height / min_height

    # 3. Iterar sobre el resto de los bboxes para agruparlos.
    for i in range(1, len(sorted_bboxes)):
        poly_bbox = sorted_bboxes[i]
        
        # Comprobar si el bbox actual se solapa con la línea actual
        if get_vertical_overlap(current_line_bbox, poly_bbox) > overlap_threshold:
            # Si pertenece a la línea, añadir su índice original y expandir el BBox de la línea
            current_line_indices.append(sorted_original_indices[i])
            current_line_bbox[0] = min(current_line_bbox[0], poly_bbox[0])
            current_line_bbox[1] = min(current_line_bbox[1], poly_bbox[1])
            current_line_bbox[2] = max(current_line_bbox[2], poly_bbox[2])
            current_line_bbox[3] = max(current_line_bbox[3], poly_bbox[3])
        else:
            # Si no pertenece, la línea anterior está completa. La guardamos.
            line_groups.append(np.array(current_line_indices))
            
            # Iniciar una nueva línea con el bbox actual
            current_line_indices = [sorted_original_indices[i]]
            current_line_bbox = poly_bbox.copy()

    # 4. Guardar la última línea que quedó en el bucle.
    if current_line_indices:
        line_groups.append(np.array(current_line_indices))

    return line_groups
    