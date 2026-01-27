# PerfectOCR/core/utils/image_analizer.py
import cv2
import matplotlib.pyplot as plt
import numpy as np
import logging
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

    # Filtrar el contorno del fondo (índice 0) desde el inicio
    for i, cont in enumerate(contours):
        cont_coords = cont.reshape(-1, 2).astype(np.int32)
        if len(cont_coords) < 4:
            continue

        cont_coords_list.append((i, cont_coords))

    if not cont_coords_list:
        return [], np.empty((0, 5))

    areas = np.array([cv2.contourArea(c[1]) for c in cont_coords_list])

    valid_mask = (areas > 1)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return [], np.empty((0, 5))

    rects = [cv2.minAreaRect(cont_coords_list[i][1]) for i in valid_indices]
    shapes = np.array([r[1] for r in rects])
    angles = np.array([r[2] for r in rects])
    valid_areas = areas[valid_indices]

    # Agrega el índice secuencial como primera columna
    metrics_array = np.column_stack([
        np.arange(len(valid_indices), dtype=np.int16),
        valid_areas,
        shapes[:, 0], # w
        shapes[:, 1], # h
        angles
    ])

    # logger.info(f"{metrics_array[:, [0, 1]]}")

    valid_coords: List[Tuple[int, np.ndarray[Any, np.dtype[np.int32]]]] = [(i, cont_coords_list[valid_indices[i]][1]) for i in range(len(valid_indices))]

    logger.info(f"Numero de contornos válidos: {len(valid_indices)}")

    contours = len(valid_coords) 
    matrix_size = metrics_array.shape[0]
    if contours != matrix_size:
        logger.info(f"Contornos dispares: {contours} != {matrix_size}")

    return valid_coords, metrics_array

def extract_contours_histogram(img: np.ndarray[Any, np.dtype[np.uint8]]):
    """
    Calcula histograma de áreas de CC.
    Retorna:
        bin_edges: edges del histograma de áreas
    """
    try:
        cont_coords, metrics = extract_contours_metrics(img)
        
        logger.debug(f"Cant coordendas: {len(cont_coords)}, metricas: {metrics.shape}")    
        biggest = np.max(metrics[:, 1])
        logger.debug(f"1: {biggest}")
        metrics = np.compress((metrics[:, 1] < biggest), metrics, 0)

        wind_lng = 3
        centroid = np.ravel((wind_lng - 1) / 2).astype(np.uint8) if wind_lng > 2 else 3
        win = np.zeros(wind_lng, np.uint8)
        windw_it = win.copy()
        np.put(windw_it,wind_lng - 1, 1)
        np.put(win, centroid, 1)
        securutyy_window = np.pad(win, 2, mode="edge",)

        # Bucle recursivo/iterativo
        while True:
            # Recalcular histograma y ventana
            hist, bin_edges = calculate_hist(metrics[:, 1])
            plt.hist(metrics[:, 1], bins='fd')  # arguments are passed to np.histogram
            plt.title("Histogram with 'auto' bins")
            (0.5, 1.0, "Histogram with 'auto' bins")
            plt.show()

            elemnts_range = hist.shape[0]
            
            # Extraer ventana de los últimos bins
            roll_window = np.take_along_axis(hist, np.arange(elemnts_range - wind_lng, elemnts_range), 0)
            
            # Verificar condiciones en orden de prioridad
            if np.array_equal(roll_window, windw_it):
                # Condición 1: Coincide con ventana Eliminar bin más grande y sus contornos
                max_area = np.max(metrics[:, 1])
                mask = metrics[:, 1] < max_area
                metrics = np.compress(mask, metrics, 0)
                hist, bin_edges = calculate_hist(metrics[:, 1])
                logger.debug(f"Eliminando por ventana fija: {hist}")
                continue  # Continuar iteración
                
            elif hist[elemnts_range - 1] == 1:
                # Condición 2: Último dígito es 1
                hist, bin_edges = calculate_hist(metrics[:, 1])
                logger.debug(f"Eliminando por ultimo dígito: {hist}")
                max_area = np.max(metrics[:, 1])
                mask = metrics[:, 1] < max_area
                metrics = np.compress(mask, metrics, 0)
                continue
                
            else:
                # Condición 3: Buscar nonzero==1 más cercano (del más grande al más pequeño)
                found_nonzero_one = False
                for bin_idx in range(elemnts_range - 1, -1, -1):  # Del más grande al más pequeño
                    if hist[bin_idx] >= 2:
                        # Si encuentra >= 2, hacer break inmediatamente
                        logger.debug("Encontrado valor >= 2, finalizando correcciones")
                        break  # Break del while True
                        
                    elif hist[bin_idx] == 1:
                        # Encontró nonzero==1 más cercano
                        # mask = np.zeros(securutyy_window.size)
                        mask = np.take_along_axis(hist, np.arange(bin_idx - wind_lng, bin_idx + wind_lng), 0)
                        if np.array_equal(securutyy_window, mask):
                            found_nonzero_one = True
                            logger.debug(f"Eliminando bin {bin_idx} con valor 1: {hist}")
                            # Eliminar contornos de este bin
                            bin_start = bin_edges[bin_idx]
                            bin_end = bin_edges[bin_idx + 1]
                            mask = ~((metrics[:, 1] >= bin_start) & (metrics[:, 1] < bin_end))
                            metrics = np.compress(mask, metrics, 0)
                            hist, bin_edges = calculate_hist(metrics[:, 1])
                            break  # Break del for, continuar while
                        
                        elif not found_nonzero_one:
                            continue
                        
                if not found_nonzero_one or hist[bin_idx] >= 2:
                    # No encontró más bins para eliminar o encontró >= 2
                    break  # Salir del while True
        return cont_coords, metrics, bin_edges
    except Exception as e:
        logger.warning(f"Error calculando histograma: {e}", exc_info=True)
        return cont_coords, metrics, bin_edges
