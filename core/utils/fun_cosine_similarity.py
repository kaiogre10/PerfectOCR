import numpy as np
from typing import List, Any, Optional
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
    return cosine_similarity(X, Y, dense_output)

