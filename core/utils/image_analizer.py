# PerfectOCR/core/utils/image_analizer.py
import cv2
# import matplotlib.pyplot as plt
import numpy as np
import logging
import time
from typing import Any, List, Tuple
from core.utils.image_utils import binarice_img

logger = logging.getLogger(__name__)

def extract_contours_metrics(img: np.ndarray[Any, np.dtype[np.uint8]], histogram: bool = False) -> Tuple[List[Tuple[int, np.ndarray[Any, np.dtype[np.int32]]]], np.ndarray[Any, Any]]:
    """
    Calcula métricas de CC robustas, filtrando ruido (rayones, manchas)
    usando Área y Solidez.
    bin_img: np.uint8, foreground=255, background=0

    Retorna:
        cont_coords: Lista de [idx_original, coords_array] para cada contorno válido
        metrics: np.ndarray con columnas [idx_original, area, width, height, angle]
    """
    bin_img = binarice_img(img, {})

    contours, _ = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return [], np.empty((0, 5))
    
    cont_coords_list: List[Tuple[int, np.ndarray[Any, np.dtype[np.int32]]]] = []
    
    sh, sw = bin_img.shape[:2]
    for i, cont in enumerate(contours):
        cont_coords = cont.reshape(-1, 2).astype(np.int32)
        if len(cont_coords) < 3:
            continue

        # x, y = cont_coords[:, 0], cont_coords[:, 1]
        # if np.any(x < 5) or np.any(x > sw-5) or np.any(y < 5) or np.any(y > sh-5):
        #     continue

        cont_coords_list.append((i, cont_coords))

        if not cont_coords_list:
            return [], np.empty((0, 5))
    
    areas = np.array([cv2.contourArea(c[1]) for c in cont_coords_list])
    
    # top_areas_list = np.sort(areas)[::-1]
    # logger.info(f"{top_areas_list[:10]}")
    
    valid_mask = (areas > 1) #& (areas < np.max(areas))

    # top_areas = np.sort(areas[valid_mask])[::-1]
    # logger.info(f"Top áreas: {top_areas[:10]}")
    
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return [], np.empty((0, 9))  # Ahora son 9 columnas
    
    pixels_val: List[int] = []
    lonely: List[int] = []
    # Máscara reutilizable
    single_mask = np.zeros((sh, sw), dtype=np.uint8)
    
    # Variables para limpiar la región anterior
    prev_x, prev_y, prev_w, prev_h = 0, 0, 0, 0
    
    # Solo procesa los contornos válidos
    hull: List[np.ndarray[Any, np.dtype[np.int32]]] = []
    for idx in valid_indices:
        conts = cont_coords_list[idx]
        
        # Limpia solo la región usada del contorno anterior
        if prev_w > 0 and prev_h > 0:
            single_mask[prev_y:prev_y+prev_h, prev_x:prev_x+prev_w] = 0
        
        convex_hull = cv2.convexHull(conts[1])
        # Calcula bounding box del contorno actual
        x, y, w, h = cv2.boundingRect(convex_hull)
        hull.append(convex_hull)
        
        # Dibuja el contorno actual
        cv2.drawContours(single_mask, [conts[1]], -1, [255], cv2.FILLED)
        
        # Extrae solo la región de interés
        roi_mask = single_mask[y:y+h, x:x+w]
        roi_img = bin_img[y:y+h, x:x+w]
        pixels = roi_img[roi_mask == 255]
        pixels_outside = roi_img[roi_mask == 0]
        
        # Verifica si HAY TINTA fuera del contorno (otro blob cerca)
        if np.count_nonzero(pixels_outside) > 0:
            lonely.append(0)  # Hay otro blob cerca (NO está solo)
        else:
            lonely.append(1)  # Está solo (sin otros blobs en el bbox)

        # 1 = nwgro (tinta), 0 = blamco (fondo)
        if np.all(pixels==255):
            pixels_val.append(1)

        else:
            pixels_val.append(0)

        prev_x, prev_y, prev_w, prev_h = x, y, w, h
        
    convex_area = np.array([cv2.contourArea(convex_hull[1]) for convex_hull in hull])
    lonely_array = np.array(lonely, dtype=np.int32)
    pixels_val_array = np.array(pixels_val, dtype=np.int32)
    black = np.count_nonzero(pixels_val_array)
    # logger.info(f"BLOBS negros: {black}, BLANCOS: {pixels_val_array.size - black}")
    # mask_log = lonely_array == 0
    # logger.info(f"Blobs solitarios: {lonely_array[mask_log].astype(np.int32)}")

    rects = [cv2.minAreaRect(cont_coords_list[i][1]) for i in valid_indices]
    shapes = np.array([r[1] for r in rects])
    angles = np.array([r[2] for r in rects])
    valid_areas = areas[valid_indices]

    centroids = np.array([(m["m10"] / m["m00"] if m["m00"] != 0 else 0, m["m01"] / m["m00"] if m["m00"] != 0 else 0)
        for m in [cv2.moments(cont_coords_list[i][1]) for i in valid_indices]], np.intp)
    
    # Agrega el índice secuencial como primera columna y pixels_val al final
    metrics_array = np.column_stack([
        np.arange(len(valid_indices), dtype=np.int32),  # 0
        valid_areas,                                    # 1
        shapes[:, 0],                                   # 2
        shapes[:, 1],                                   # 3
        angles,                                         # 4
        centroids[:, 0],                                # 5
        centroids[:, 1],                                # 6
        convex_area,                                    # 7
        pixels_val_array,                               # 8
        lonely_array                                    # 9
    ])

    if histogram:
        metrics_array = extract_contours_histogram(metrics_array)

    filtered_original_indices = metrics_array[:, 0].astype(np.int32)
    valid_coords: List[Tuple[int, np.ndarray[Any, np.dtype[np.int32]]]] = [(int(idx), cont_coords_list[valid_indices[int(idx)]][1]) for idx in filtered_original_indices]

    valid_contours = len(valid_coords) 
    matrix_size = metrics_array.shape[0]
    if valid_contours != matrix_size:
        logger.warning(f"Contornos dispares: {valid_contours} != {matrix_size}")
        return []

    logger.info(f"Contornos validos: {valid_contours}")

    return valid_coords, metrics_array

def extract_contours_histogram(metrics: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """
    Calcula histograma de áreas de contornos.
    Retorna:
        bin_edges: edges del histograma de áreas
    """
    time_h = time.perf_counter()
    areas = metrics[:, 1]
    top_area = np.max(areas) + 1
    hist, bin_edges = np.histogram(areas, bins=(np.histogram_bin_edges(areas, 'fd', (0.0, top_area))).astype(np.float32))

    hist_rever = hist[::-1].astype(np.int32)
    cutting = np.where(hist_rever > 2)[0]
    idx_orig = len(hist) - 1 - cutting[0] if cutting.size > 0 else -1
    outliers_indx = np.nonzero(hist==1)[0]
    filtered_outliers = outliers_indx[outliers_indx > idx_orig]
    if filtered_outliers.size == 0:
        logger.warning("Imagen sin outliers")
        return metrics
    
    # logger.info(f"HIST 1: {hist.shape}")
    mask = np.min(filtered_outliers) - 1
    ind_big = bin_edges[mask] 
    cond = areas < ind_big
    metrics = np.compress(cond, metrics, 0)
    areas = metrics[:, 1]
    top_area = np.max(areas) + 1
    hist, bin_edges = np.histogram(areas, bins=(np.histogram_bin_edges(areas, 'fd', (0.0, top_area))).astype(np.float32))
    # plt.hist(metrics[:, 1], bins='fd')  # arguments are passed to np.histogram
    # plt.title("Histogram with 'fd' bins")
    # (0.5, 1.0, "Histogram with 'fd' bins")
    # plt.show()
    # logger.info(f"HIST 2: {hist.shape}")
    logger.debug(f"Analisis de histograma completado en {time.perf_counter()-time_h}'s")
    return metrics
    
def extract_cc_metrics(bin_img: np.ndarray[Any, np.dtype[np.uint8]], mask_contours):
    
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_contours, connectivity=8)

    mask_cc = np.zeros((bin_img.shape[0], bin_img.shape[1]), np.uint8)
    
    for label in range(1, n_labels):  # Salta background (0)
        area = stats[label, cv2.CC_STAT_AREA]
        
        # ===== PASO 3: Filtrar CC por área (elimina picos aislados) =====
        if area > min_area_cc:
            # Obtén píxeles de esta componente
            component_mask = (labels == label).astype(np.uint8) * 255
            mask_cc |= component_mask
    
    # ===== PASO 4: Rellena agujeros internos pequeños en caracteres =====
    # Dilate + erode para cerrar pequeños agujeros
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_filled = cv2.morphologyEx(mask_cc, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    logger.info(f"CC válidas: {n_labels - 1} → {len(np.unique(labels[mask_filled > 0])) - 1}")
    
    return mask_filled

    label_labeled = np.arange(1, n_labels).astype(np.uint32)
    cx, cy = np.hsplit(centroids[1:], 2)
    mapped_stats = np.column_stack([label_labeled, stats[1:, cv2.CC_STAT_AREA], stats[1:, cv2.CC_STAT_LEFT], stats[1:, cv2.CC_STAT_TOP], stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT], cx, cy]).astype(np.uint32)
    logger.info(f"MAPPED_STATS: {mapped_stats}") 
    # logger.info(f"LABELS: {labels[1:].shape}, CENTROIDS: {centroids[:-1].shape}, N LABLES: {n_labels-1}")
    # logger.info(f"Tmaño imagne: '{bin_img.shape}'")
    # unique_labs = np.unique(labels, return_index=True, axis=0)[0]

    # logger.info(f"LABELS: {unique_labs}")
    # extract_labs = np.extract(unique_labs, labels)
    # logger.info(f"{extract_labs}")

def get_all_morph(img: np.ndarray[Any, np.dtype[np.uint8]]):
    bin_img = binarice_img(img, {})
    valid_coords, _ = extract_contours_metrics(bin_img)

    mask_contours = np.zeros((bin_img.shape[0], bin_img.shape[1]), np.uint8)

    for _, cont in valid_coords:
        cv2.drawContours(mask_contours, [cont], -1, [255], cv2.FILLED)
    
    mask_filled = extract_cc_metrics(bin_img, mask_contours)
    return mask_filled
