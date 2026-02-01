# PerfectOCR/core/utils/image_analizer.py
import cv2
# import matplotlib.pyplot as plt
import numpy as np
import logging
import time
from typing import Any, List, Tuple
from core.utils.image_utils import binarice_img
from core.utils.math_utils import calculate_hist

logger = logging.getLogger(__name__)

def extract_contours_metrics(img: np.ndarray[Any, np.dtype[np.uint8]]) -> Tuple[List[Tuple[int, np.ndarray[Any, np.dtype[np.int32]]]], np.ndarray[Any, Any]]:
    """
    Calcula métricas de CC robustas, filtrando ruido (rayones, manchas)
    usando Área y Solidez.
    bin_img: np.uint8, foreground=255, background=0

    Retorna:
        cont_coords: Lista de [idx_original, coords_array] para cada contorno válido
        metrics: np.ndarray con columnas [idx_original, area, width, height, angle]
    """
    bin_img = binarice_img(img, {})
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return [], np.empty((0, 5))

    cont_coords_list: List[Tuple[int, np.ndarray[Any, np.dtype[np.int32]]]] = []

    for i, cont in enumerate(contours):
        cont_coords = cont.reshape(-1, 2).astype(np.int32)
        if len(cont_coords) < 3:
            continue

        cont_coords_list.append((i, cont_coords))

    if not cont_coords_list:
        return [], np.empty((0, 5))

    areas = np.array([cv2.contourArea(c[1]) for c in cont_coords_list])

    valid_mask = (areas > 0) & (areas != np.max(areas))
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return [], np.empty((0, 5))

    rects = [cv2.minAreaRect(cont_coords_list[i][1]) for i in valid_indices]
    shapes = np.array([r[1] for r in rects])
    angles = np.array([r[2] for r in rects])
    valid_areas = areas[valid_indices]
    
    centroids = np.array([(m["m10"] / m["m00"] if m["m00"] != 0 else 0, m["m01"] / m["m00"] if m["m00"] != 0 else 0)
        for m in [cv2.moments(cont_coords_list[i][1]) for i in valid_indices]], np.intp)
    
    # Agrega el índice secuencial como primera columna
    metrics_array = np.column_stack([
        np.arange(len(valid_indices), dtype=np.int32),
        valid_areas,
        shapes[:, 0], # w
        shapes[:, 1], # h
        angles,
        centroids[:, 0],
        centroids[:, 1]
    ])

    valid_coords: List[Tuple[int, np.ndarray[Any, np.dtype[np.int32]]]] = [(i, cont_coords_list[valid_indices[i]][1]) for i in range(len(valid_indices))]

    logger.debug(f"Numero de contornos válidos: {len(valid_indices)}")

    contours = len(valid_coords) 
    matrix_size = metrics_array.shape[0]
    if contours != matrix_size:
        logger.info(f"Contornos dispares: {contours} != {matrix_size}")

    return valid_coords, metrics_array

def extract_contours_histogram(img: np.ndarray[Any, np.dtype[np.uint8]]):
    """
    Calcula histograma de áreas de contornos.
    Retorna:
        bin_edges: edges del histograma de áreas
    """
    time_h = time.perf_counter()

    cont_coords, metrics = extract_contours_metrics(img)
    biggest = np.max(metrics[:, 1])
    logger.debug(f"1: {biggest}")
    metrics = np.compress((metrics[:, 1] < biggest), metrics, 0)    
    hist, bin_edges = calculate_hist(metrics[:, 1])
    
    ouliers_indx = np.nonzero(hist==1)[0]
    mask = np.min(ouliers_indx)
    ind_big = bin_edges[mask]
    cond = metrics[:, 1] < ind_big
    metrics = np.compress(cond, metrics, 0)
    hist, bin_edges = calculate_hist(metrics[:, 1])    
    # plt.hist(metrics[:, 1], bins='fd')  # arguments are passed to np.histogram
    # plt.title("Histogram with 'auto' bins")
    # (0.5, 1.0, "Histogram with 'auto' bins")
    # plt.show()
                
    logger.debug(f"Analisis de histograma completado en {time.perf_counter()-time_h}'s")
    return cont_coords, metrics, bin_edges
        