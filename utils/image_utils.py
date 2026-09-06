# PerfectOCR/core/utils/image_utils.py
import cv2
import numpy as np
from typing import Any, Optional, List, Dict, Tuple
import logging
from skimage.filters import threshold_sauvola, unsharp_mask #type: ignore
import time
from core.assets.assets import WHITE

_white = WHITE

logger = logging.getLogger(__name__)

def make_contiguous(img_arr: np.ndarray[Any, Any]) -> np.ndarray[Any, np.dtype[np.uint8]]:
    return img_arr if img_arr.flags.c_contiguous else np.ascontiguousarray(img_arr, dtype=np.uint8)

def normalice_image(img: Optional[np.ndarray[Any, Any]]) -> Optional[np.ndarray[Any, np.dtype[np.uint8]]]:
    """
    - Convierte BGR->GRAY si viene con 3/4 canales
    - Convierte a dtype uint8 (escala floats en [0,1] a 0-255)
    - Garantiza que el array sea C-contiguo
    Retorna ndarray uint8 o None si ocurre un error / imagen vacía.
    """
    try:
        if img is None:
            logger.error("normalice_image: imagen None recibida")
            return None
        
        # Si llega en color, convertir a gris (BGR->GRAY)
        dims = img.ndim

        if dims == 2:
            channels = 1
        elif dims == 3:
            channels = img.shape[2]
        else:
            return None
        
        if channels > 2:
            image_arr = decolorate(img)
        
            if image_arr.size < 2:
                return None
        
            elif channels == 4:
                img_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGRA2GRAY)

            else:
                img_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)
        
        elif channels == 2:
            img_arr = img[:, :, 0]
        
        else:  # channels == 1
            img_arr = img
            
        if img_arr.dtype != np.uint8:
            if np.issubdtype(img_arr.dtype, np.floating):
                mx = float(img_arr.max()) if img_arr.size > 0 else 0.0
                if mx <= 1.0:
                    img_arr = (img_arr * 255.0).round().astype(np.uint8, copy=False)
                    
                else:
                    img_arr = np.clip(img_arr, 0, 255).round().astype(np.uint8, copy=False)
                
            else:
                img_arr = img_arr.astype(np.uint8, copy=False)
                
        if not validate_image(img_arr):
            return None
            
        return np.require(img_arr, dtype=np.uint8, requirements=['C', 'A', 'W', 'O', 'E'])
        
    except Exception  as e:
        logger.error(f"Error normalizando imagen: {e}", exc_info=True)
    return None

def elevate_dims(image_list: List[np.ndarray[Any, Any]]) -> List[np.ndarray[Any, Any]]:
    try:
        return [make_contiguous(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)) for img in image_list]
    except cv2.error as e:
        logger.critical(f"Error añadiendo dimensiones a la imagen: {e}", exc_info=True)
    return []

def validate_image(img: Optional[np.ndarray[Any, Any]]):
    return False if img is None else (7.0 < np.mean(img) < 251.0)

def use_bilateral_filter(img: np.ndarray[Any, np.dtype[np.uint8]], d: int, sigma_color: int, sigma_space: int)-> np.ndarray[Any, np.dtype[np.uint8]]:
    return make_contiguous(cv2.bilateralFilter(img, d, sigma_color, sigma_space))

def use_sobel(img: np.ndarray[Any, np.dtype[np.uint8]], ksize: int):
    return np.mean(np.abs(cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=ksize)), dtype=np.float32)

def get_rotation_matrix(center: Tuple[int, int], angle: float):
    return cv2.getRotationMatrix2D(center, angle, 1.0)

def get_image_lines(full_img: np.ndarray[Any, Any], canny_thresholds: Tuple[int, int], hough_threshold: int, min_len: int, hough_max_line_gap_px: int):
    edges = cv2.Canny(full_img, canny_thresholds[0], canny_thresholds[1])
    return cv2.HoughLinesP(edges, 1, np.pi/180, threshold=hough_threshold, minLineLength=min_len, maxLineGap=hough_max_line_gap_px)

def rotate_matrix(full_img: np.ndarray[Any, Any], rotation_matrix: np.ndarray[Any, Any], new_w: int, new_h: int):
    return cv2.warpAffine(full_img, rotation_matrix, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=_white)

def binarice_img(cropped_img: np.ndarray[Any, np.dtype[np.uint8]], worker_config: Dict[str, Any]) -> np.ndarray[Any, np.dtype[np.uint8]]:
    if is_binarized(cropped_img):
        return make_contiguous(cropped_img)
        
    c_value: int = worker_config.get('c_value', 7)
    height_thresholds: List[int] = worker_config.get('height_thresholds_px', [100, 800, 1500, 2500])
    block_sizes_map: List[int] = worker_config.get('block_sizes_map', [15, 21, 25, 35, 41])
    height = int(cropped_img.shape[0])

    block = get_adaptive_block_size(height, height_thresholds, block_sizes_map)
    mode: str = measure_polygon_quality(cropped_img)

    cropped_img = make_contiguous(cropped_img)

    if mode == "otsu":
        bin_img = otsu_binarize(cropped_img)
    elif mode == "adaptive_gaussian":
        bin_img = adaptive_binarize(cropped_img, block, c_value)
    elif mode == "sauvola":
        bin_img = sauvola_binarize(cropped_img, block)
    else:
        bin_img = adaptive_mean_fallback(cropped_img, block, c_value)

    return make_contiguous(cv2.bitwise_not(bin_img))
   
def get_adaptive_block_size(height: float, height_thresholds: List[int], block_sizes_map: List[int]) -> int:
    """Calcula el tamaño de bloque adaptativo basado en la altura del polígono."""
    for i, threshold in enumerate(height_thresholds):
        if height < threshold:
            block_size = block_sizes_map[min(i, len(block_sizes_map) - 1)]
            return max(3, block_size if block_size % 2 != 0 else block_size + 1)
    final_block_size = block_sizes_map[-1]
    return max(3, final_block_size if final_block_size % 2 != 0 else final_block_size + 1)

def measure_polygon_quality(cropped_img: np.ndarray[Any, np.dtype[np.uint8]]) -> str:
    """
    Analiza la imagen en escala de grises (histograma, std) para
    decidir el mejor método de binarización.
    """
    std = np.std(cropped_img, dtype=np.float32)
    if std == 0: return "adaptive_mean"
    hist = cv2.calcHist([cropped_img], [0], None, [255], [0, 255]).flatten()
    peaks = np.sum((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:]), dtype=np.float32)
    prob = hist / np.sum(hist, dtype=np.float32)
    entropy = -np.sum(prob * np.log2(prob + 1e-8))

    if peaks > 1.0 and std > 30.0:
        return "otsu"  # Alto contraste, bimodal

    elif std > 20.0 and entropy > 5.0:
        return "adaptive_gaussian"  # Contraste variable

    elif std > 10.0:
        return "sauvola"  # Texto sobre fondo no uniforme

    else:
        return "adaptive_mean"  # Bajo contraste, imagen "plana"

def otsu_binarize(cropped_img: np.ndarray[Any, np.dtype[np.uint8]]) -> np.ndarray[Any, np.dtype[np.uint8]]:
    resultis = cv2.threshold(cropped_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return make_contiguous(resultis[1])

def adaptive_binarize(cropped_img: np.ndarray[Any, np.dtype[np.uint8]], block_size: int, c_value: int) -> np.ndarray[Any, np.dtype[np.uint8]]:
    return make_contiguous(cv2.adaptiveThreshold(cropped_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c_value))

def sauvola_binarize(cropped_img: np.ndarray[Any, np.dtype[np.uint8]], adaptive_block_size: int) -> np.ndarray[Any, np.dtype[np.uint8]]:
    """Sauvola thresholding producing uint8 mask with text as foreground (0) and background (255)"""
    thresh_sauvola = threshold_sauvola(image=cropped_img, window_size=adaptive_block_size)
    bin_bool = (cropped_img > thresh_sauvola)
    return make_contiguous(bin_bool * 255)

def apply_clahe_correction(original_img: np.ndarray[Any, np.dtype[np.uint8]], clip_limit: float, grid_size: Tuple[ Any, ...]) -> np.ndarray[Any, Any]:
    """Aplica el filtro CLAHE a una imagen."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(original_img)

def apply_sharpening_correction(cropped_img_np: np.ndarray[Any, np.dtype[np.uint8]], radius: float, amount: float) -> np.ndarray[Any, np.dtype[np.uint8]]:
    """Aplica el filtro unsharp_mask a una imagen."""
    sharpened_float = unsharp_mask(cropped_img_np, radius=radius, amount=amount)
    # unsharp_mask devuelve un float en [0, 1], se debe convertir de vuelta a uint8 [0, 255]
    return make_contiguous(np.clip(sharpened_float, 0, 1) * 255)

def adaptive_mean_fallback(cropped_img: np.ndarray[Any, np.dtype[np.uint8]], block_size: int, c_value: int) -> np.ndarray[Any, np.dtype[np.uint8]]:
    return make_contiguous(cv2.adaptiveThreshold(cropped_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, max(1, c_value - 2)))

def decolorate(full_img: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Elimina colores (rayones, resaltados, etc.) de la imagen, dejando solo blanco y negro."""
    mask_black = np.all(full_img[:, :, :3] < 160, axis=2)
    mask_white = np.all(full_img[:, :, :3] > 180, axis=2)
    mask_valid = mask_black | mask_white

    fill = [255, 255, 255, 255] if full_img.shape[2] == 4 else _white
    full_img[~mask_valid] = fill
    
    return full_img

# def get_conected_comps_metrics(img: np.ndarray[Any, np.dtype[np.uint8]]):
#     # img_bin: imagen binaria (uint8, valores 0/255)
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=4)

#     # Visualizar: cada componente con color aleatorio
#     h, w = labels.shape
#     vis = np.zeros((h, w, 3), dtype=np.uint8)

#     # label 0 es fondo, por eso empezamos en 1
#     rng = np.random.default_rng(42)
#     colors = rng.integers(0, 255, size=(num_labels, 3), dtype=np.uint8)
#     colors[0] = [0, 0, 0]

#     for label_id in range(1, num_labels):
#         vis[labels == label_id] = colors[label_id]

#     # (Opcional) dibujar bounding boxes y centroides
#     for label_id in range(1, num_labels):
#         x, y, bw, bh, area = stats[label_id]
#         cx, cy = centroids[label_id]
#         cv2.rectangle(vis, (x, y), (x + bw, y + bh), (255, 255, 255), 1)
#         cv2.circle(vis, (int(cx), int(cy)), 2, (0, 255, 255), -1)

#     screen_w, screen_h = 1366, 768 
#     h, w = vis.shape[:2]
#     scale = min(screen_w / w, screen_h / h, 1.0)
#     show_w, show_h = int(w * scale), int(h * scale)
#     cv2.namedWindow("Componentes conectados", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("Componentes conectados", show_w, show_h)
#     cv2.imshow("Componentes conectados", vis)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

def get_contours_values(img: np.ndarray[Any, np.dtype[np.uint8]]) -> Tuple[List[Tuple[int, np.ndarray[Any, np.dtype[np.int32]]]], np.ndarray[Any, Any]]:
    """Calcula UNICAMENTE los features de OPEN CV"""
    time0 = time.perf_counter()
    bin_img = binarice_img(img, {})

    contours_hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_hierarchy[0]
    if not contours:
        return [], np.empty(0)
    
    total_conts = len(contours)
    contours = [np.array(cont.reshape(-1, 2), np.int32) for cont in contours]
    # logger.info(f"{total_conts} Contornos encontrados")

    cont_coords_list: List[Tuple[int, np.ndarray[Any, np.dtype[np.int32]]]] = []
    cols = 16
    metrics_array_new = np.zeros((total_conts, cols), dtype=np.float32, order='F')
    # single_mask = bin_img.copy()
    # single_mask[:] = 0

    for i, cont_coords in enumerate(contours):
        # for c in range(cols):
        if len(cont_coords) < 3:
            continue
        area = cv2.contourArea(cont_coords)
        if area < 2:
            continue

        metrics_array_new[i, 0] = area
        centroids, dims, angle = cv2.minAreaRect(cont_coords)
        metrics_array_new[i, [1, 2]] = centroids
        metrics_array_new[i, [3, 4]] = dims
        metrics_array_new[i, 5] = angle
        
        conv_hull = cv2.convexHull(cont_coords)
        metrics_array_new[i, 6] = cv2.arcLength(cont_coords, True)  # Perimetro contorno
        metrics_array_new[i, 7] = cv2.contourArea(conv_hull)        # Area de convexo
        metrics_array_new[i, 8] = cv2.arcLength(conv_hull, True)    # Permietro del convex

        x, y, w, h = cv2.boundingRect(cont_coords)
        metrics_array_new[i, 9] = x                                 # Ancho del bbox de contorno
        metrics_array_new[i, 10] = y                                # Alto del bbox contorno
        metrics_array_new[i, 11] = w
        metrics_array_new[i, 12] = h

        roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(roi_mask, [cont_coords], -1, [255], cv2.FILLED, offset=(-x, -y))
        roi_img = bin_img[y:y+h, x:x+w]
        black_pixels = cv2.countNonZero(cv2.bitwise_and(roi_img, roi_mask))     # Total píxeles negros dentro contorno
        inside = roi_img[roi_mask == 255]                                       # pixeles dentro del contorno
        total_inside = inside.size                                              # total píxeles del contorno
        
        metrics_array_new[i, -2] = black_pixels
        metrics_array_new[i, -1] = total_inside
        
        # logger.info(f"{internal_pixels}, {density}")

        cont_coords_list.append((i, cont_coords))

    if not cont_coords_list or metrics_array_new.size == 0:
        return [], np.empty(0)

    hierarchy = contours_hierarchy[1]
    h = hierarchy[0]  # shape: (N, 4) => [next, prev, first_child, parent]

    childs = (h[:, 2] != -1)  # tiene hijo directo
    is_outer = (h[:, 3] == -1)  # es contorno exterior
    is_hollow_outer = childs & is_outer  # letra/figura exterior con hueco
    is_solid_outer = (~childs) & is_outer  # letra/figura exterior sólida
    idx = np.arange(total_conts, dtype=np.int16)

    contours_features_array = np.column_stack([idx, metrics_array_new, is_outer[idx], is_hollow_outer[idx], is_solid_outer[idx]])
    contours_features_array = contours_features_array[[c[0] for c in cont_coords_list]]
    
    # logger.info(f"contours_features_array SHAPE: {contours_features_array.shape} ARRAY:\n"f"{np.array2string(contours_features_array, suppress_small=True)}")
    valid_coords = cont_coords_list
    valid_contours = len(valid_coords)
    matrix_size = contours_features_array.shape[0]
    if valid_contours != matrix_size:
        logger.warning(f"Contornos dispares: {valid_contours} != {matrix_size}")
        return [], np.empty((0, 5))
        
    logger.info(f"Tiempo calculando features de {valid_contours} contornos: {time.perf_counter()-time0:.6f}'s")
    return cont_coords_list, contours_features_array

# def calculate_complementary_feats(metrics_array_new: np.ndarray[Any, Any]):
#     rec_widith = metrics_array_new[:, 1]
#     rec_height = metrics_array_new[:, 2]
#     rect_area = rec_widith * rec_height
#     angles = metrics_array_new[:, 3]
#     convex_area = metrics_array_new[:, 6]
#     # Para cada rectángulo, tomamos como min_side el lado (ancho/alto) más perpendicular/vertical al eje Y, es decir, el que corresponde al ángulo más cercano a 90°
#     min_side = np.where(metrics_array_new[:, 3] > 45, rec_widith, rec_height)
#
#     angle_norm = np.where(rec_widith < rec_height, angles + 90, angles)
#     angle_norm = angle_norm % 180.0
#     angles = angle_norm
#
#     ratio_areas = convex_area / rect_area
#     aspect_ratio = (np.maximum(rec_widith, rec_height) / np.minimum(rec_widith, rec_height))
#
#     irregular_ratio = utils_array[:, 0] / utils_array[:, 1] # COLUMNA 16
#
#     metrics_array = np.column_stack([
#         idx,                        # COLUMNA 0: Índice original del contorno
#         metrics_array_new[:, 0],    # COLUMNA 1: Área contorno
#         rec_widith,                 # COLUMNA 2: Ancho del rectángulo mínimo
#         rec_height,                 # COLUMNA 3: Alto del rectángulo mínimo
#         angles,                     # COLUMNA 4: Ángulo del rectángulo mínimo
#         metrics_array_new[:, 4],    # COLUMNA 5: Centroide X (contorno)
#         metrics_array_new[:, 5],    # COLUMNA 6: Centroide Y (contorno)
#         convex_area,                # COLUMNA 7: Área del polígono convexo
#         aspect_ratio,               # COLUMNA 8: Relación de aspecto (mayor/menor lado)
#         childs,                     # COLUMNA 9: Tiene hijos (bool)
#         ratio_areas,                # COLUMNA 10: Área convexa / Área min Rect
#         rect_area,                  # COLUMNA 11: Área del Min Rect
#         irregular_ratio,            # COLUMNA 12: Relación perímetro convexo/contorno
#         min_side                    # COLUMNA 13: Lado mínimo respecto al eje Y
#     ])
#
#     metrics_array = metrics_array[[c[0] for c in cont_coords_list]]
#     valid_coords = cont_coords_list
#
#     valid_contours = len(valid_coords)
#     matrix_size = metrics_array.shape[0]
#     if valid_contours != matrix_size:
#         logger.warning(f"Contornos dispares: {valid_contours} != {matrix_size}")
#         return [], np.empty((0, 5))
#
#     # logger.info(f"METRICS LIST: {metrics_array.shape}")
#     logger.info(f"Tiempo extrayendo métricas VECTOROIZADAS: {time.perf_counter()-time0:.6f}'s")
#     return valid_coords, metrics_array

def is_binarized(img: np.ndarray[Any, Any]):
    """True si es una imagen está binarizada"""
    return np.all((img == 0) | (img == 255))

def configure_kernel(x: int, y: int):
    return cv2.getStructuringElement(cv2.MORPH_CROSS, (x, y))

def morph_operations(img: np.ndarray[Any, Any], kernel: Any, iterations: int) -> np.ndarray[Any, Any]:
    return make_contiguous(cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations))