import cv2
import numpy as np
from typing import Any, Optional, List, Dict, Tuple
import logging
from skimage.filters import threshold_sauvola #type: ignore
import time

logger = logging.getLogger(__name__)

def make_contiguous(img_arr: np.ndarray[Any, Any]) -> np.ndarray[Any, np.dtype[np.uint8]]:
    return img_arr if img_arr.flags.c_contiguous else np.ascontiguousarray(img_arr, np.uint8)

def normalice_image(img: Optional[np.ndarray[Any, Any]]) -> Optional[np.ndarray[Any, np.dtype[np.uint8]]]:
    """
    Normaliza una imagen de entrada:
    - Asegura que sea ndarray
    - Convierte BGR->GRAY si viene con 3/4 canales
    - Convierte a dtype uint8 (escala floats en [0,1] a 0-255)
    - Garantiza que el array sea C-contiguo
    Retorna ndarray uint8 o None si ocurre un error / imagen vacía.
    """
    try:
        if img is None:
            logger.error("normalice_image: imagen None recibida")
            return None
        
        if not validate_image(img):
            logger.debug("Imagen blanca/negra completamente")
            return None

        try:
            img_arr = np.asarray(img, dtype=np.uint8)
        except Exception as e:
            logger.error(f"normalice_image: no se pudo convertir a ndarray: {e}", exc_info=True)
            return None

        # Si llega en color, convertir a gris (BGR->GRAY)
        if img_arr.ndim == 3 and img_arr.shape[2] in (3, 4):
            try:
                img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
                logger.debug("normalice_image: convertida imagen BGR->GRAY")
            except cv2.error as e:
                logger.warning(f"normalice_image: no se pudo convertir BGR->GRAY: {e}", exc_info=True)

        # Asegurar dtype uint8 (flotantes se escalan si están en [0,1], otros se recortan)
        if img_arr.dtype != np.uint8:
            try:
                if np.issubdtype(img_arr.dtype, np.floating):
                    mx = float(img_arr.max()) if img_arr.size > 0 else 0.0
                    if mx <= 1.0:
                        img_arr = (img_arr * 255.0).round().astype(np.uint8)
                        
                    else:
                        img_arr = np.clip(img_arr, 0, 255).round().astype(np.uint8)
                    logger.info("normalice_image: convertida imagen float->uint8 (escalada si hizo falta)")
                    
                else:
                    img_arr = img_arr.astype(np.uint8, copy=False)
                    logger.info("normalice_image: casteada imagen a uint8")
                    
            except TypeError as e:
                logger.error(f"normalice_image: fallo al convertir dtype: {e}", exc_info=True)
                
                try:
                    img_arr = np.array(img_arr, dtype=np.uint8)
                except TypeError:
                    return None

        # 
        # Logueo detallado para trazabilidad
        # try:
        #     vmin = int(img_arr.min()); vmax = int(img_arr.max()); vmean = float(img_arr.mean())
        # except Exception:
        #     vmin = vmax = None; vmean = None

        # logger.debug(
        #     "normalice_image: id=%d shape=%s dtype=%s min=%s max=%s mean=%s",
        #     id(img_arr), getattr(img_arr, "shape", None), getattr(img_arr, "dtype", None),
        #     vmin, vmax, f"{vmean:.2f}" if vmean is not None else None
        # )
    
        return make_contiguous(img_arr)
        
    except Exception  as e:
        logger.error(f"Error normalizando imagen: {e}", exc_info=True)
    return None

def elevate_dims(image_list: List[np.ndarray[Any, Any]]) -> List[np.ndarray[Any, Any]]:
    try:
        return [make_contiguous(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)) for img in image_list]
    except cv2.error as e:
        logger.critical(f"Error añadiendo dimensiones a la imagen: {e}", exc_info=True)
    return []

def calculate_img_values(img: np.ndarray[Any, Any]):
    img_mean = np.mean(img)
    img_dims = img.shape[:2]
    return int(img_mean), img_dims

def validate_image(img: Optional[np.ndarray[Any, Any]]) -> bool:
    return bool(7 < int(np.mean(img)) < 251) if img is not None else False

def cropp_img(full_img: np.ndarray[Any, np.dtype[np.uint8]], all_bboxes: List[np.ndarray[Any, Any]] | np.ndarray[Any, Any], padding: Optional[int] = None) -> np.ndarray[Any, np.dtype[np.uint8]]:
    img_h = full_img.shape[0]
    img_w = full_img.shape[1]

    if padding is None:
        padding = 1

    bboxes_array = np.array(all_bboxes).astype(np.int16)

    # logger.info(f"{bboxes_array.shape}")

    if bboxes_array.ndim == 1 and bboxes_array.shape[0] == 4:
        bboxes_array = bboxes_array.reshape(1, 4)

    x1, y1, x2, y2 = bboxes_array[:, 0], bboxes_array[:, 1], bboxes_array[:, 2], bboxes_array[:, 3]

    valid_dims = (x2 >= x1) & (y2 >= y1)

    if not np.any(valid_dims):
        logger.warning("Dimensiones no validas")

    # Aplicar padding y clipping
    px1 = max(0, int(np.min(x1 - padding)))
    py1 = max(0, int(np.min(y1 - padding)))
    px2 = min(img_w, int(np.max(x2 + padding)))
    py2 = min(img_h, int(np.max(y2 + padding)))

    crop_x1, crop_y1 = px1, py1
    crop_x2, crop_y2 = px2, py2

    cropped: np.ndarray[Any, np.dtype[np.uint8]] = make_contiguous(full_img[crop_y1:crop_y2, crop_x1:crop_x2])
    return cropped

def use_bilateral_filter(img: np.ndarray[Any, np.dtype[np.uint8]], d: int, sigma_color: int, sigma_space: int)-> np.ndarray[Any, np.dtype[np.uint8]]:
    return make_contiguous(cv2.bilateralFilter(img, d, sigma_color, sigma_space))

def use_sobel(img: np.ndarray[Any, np.dtype[np.uint8]], ksize: int) -> float:
    sobel, _ = calculate_img_values(np.abs(cv2.Sobel(img, cv2.CV_64F, 1, 1, ksize))) #type: ignore
    return float(sobel)

def binarice_img(cropped_img: np.ndarray[Any, np.dtype[np.uint8]], worker_config: Dict[str, Any]) -> np.ndarray[Any, np.dtype[np.uint8]]:    
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
    std = np.std(cropped_img)
    if std == 0: return "adaptive_mean"
    hist = cv2.calcHist([cropped_img], [0], None, [255], [0, 255]).flatten()
    peaks = np.sum((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:]))
    prob = hist / np.sum(hist)
    entropy = -np.sum(prob * np.log2(prob + 1e-8))

    if peaks > 1 and std > 30:
        return "otsu"  # Alto contraste, bimodal

    elif std > 20 and entropy > 5.0:
        return "adaptive_gaussian"  # Contraste variable

    elif std > 10:
        return "sauvola"  # Texto sobre fondo no uniforme

    else:
        return "adaptive_mean"  # Bajo contraste, imagen "plana"

def otsu_binarize(cropped_img: np.ndarray[Any, np.dtype[np.uint8]]) -> np.ndarray[Any, np.dtype[np.uint8]]:
    resultis = cv2.threshold(cropped_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return make_contiguous(resultis[1])

def adaptive_binarize(cropped_img: np.ndarray[Any, np.dtype[np.uint8]], block_size: int, c_value: int) -> np.ndarray[Any, np.dtype[np.uint8]]:
    return make_contiguous(cv2.adaptiveThreshold(cropped_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c_value))

def sauvola_binarize(cropped_img: np.ndarray[Any, np.dtype[np.uint8]], adaptive_block_size: int) -> np.ndarray[Any, np.dtype[np.uint8]]:
    """Sauvola thresholding producin0g uint8 mask with text as foreground (0) and background (255)"""
    thresh_sauvola = threshold_sauvola(image=cropped_img, window_size=adaptive_block_size) 
    bin_bool = (cropped_img > thresh_sauvola)
    return make_contiguous(bin_bool * 255)

def adaptive_mean_fallback(cropped_img: np.ndarray[Any, np.dtype[np.uint8]], block_size: int, c_value: int) -> np.ndarray[Any, np.dtype[np.uint8]]:
    return make_contiguous(cv2.adaptiveThreshold(cropped_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, max(1, c_value - 2)))

def decolorate(full_img: np.ndarray[Any, np.dtype[np.uint8]]) -> np.ndarray[Any, np.dtype[np.uint8]]:
    """
    Elimina colores (rayones, resaltados, etc.) de la imagen, dejando solo blanco y negro.
    """
    full_img = make_contiguous(full_img)
    # Máscara para píxeles negros (todos los canales <= threshold_black)
    mask_black = np.all(full_img < 160, axis=2)
    
    # Máscara para píxeles blancos (todos los canales >= threshold_white)
    mask_white = np.all(full_img > 180, axis=2)
    
    # Máscara de píxeles válidos (negro o blanco)
    mask_valid = mask_black | mask_white
    
    # Reemplaza los píxeles de color (no válidos) por blanco
    full_img[~mask_valid] = [255, 255, 255]

    # Convierte a escala de grises para continuar el flujo normal
    gray = normalice_image(full_img)
    if not validate_image(gray):
        logger.warning("Normalice IMG devolvío imagen, Imagen en grises de cv2")
        return make_contiguous(cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY))
    else:
        return make_contiguous(cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY))

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

def vec_contours_metrics(img: np.ndarray[Any, np.dtype[np.uint8]]) -> Tuple[List[Tuple[int, np.ndarray[Any, np.dtype[np.int32]]]], np.ndarray[Any, Any]]:
    """
    Calcula métricas de CC robustas, filtrando ruido (rayones, manchas)
    usando Área y Solidez.
    bin_img: np.uint8, foreground=255, background=0

    Retorna:
        cont_coords: Lista de [idx_original, coords_array] para cada contorno válido
        metrics: np.ndarray con columnas [idx_original, area, width, height, angle]
    """
    time0 = time.perf_counter()
    if is_binarized(img):
        logger.info("Imagen ya binaria")
        bin_img = img
    else:
        bin_img = binarice_img(img, {})

    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return [], np.empty((0, 5))
    
    contours = [np.array(cont.reshape(-1, 2), np.int32) for cont in contours]
    total_conts = len(contours)
    
    cont_coords_list: List[Tuple[int, np.ndarray[Any, np.dtype[np.int32]]]] = []
    bbox_cont = np.zeros((total_conts, 4), np.int16)
    metrics_array_new = np.zeros((total_conts, 19), np.float32)

    for i, cont_coords in enumerate(contours):
        if len(cont_coords) < 3:
            continue
        
        area = cv2.contourArea(cont_coords)
        if area < 2:
            continue

        _, rec_shape, angle = cv2.minAreaRect(cont_coords)

        metrics_array_new[i, 0] = area
        metrics_array_new[i, [1, 2]] = rec_shape
        metrics_array_new[i, 3] = angle

        bbox_cont[i] = cv2.boundingRect(cont_coords)
        conv_hull = cv2.convexHull(cont_coords)
        
        metrics_array_new[i, -1] = cv2.arcLength(conv_hull, True)
        metrics_array_new[i, -2] = cv2.arcLength(cont_coords, True)
        metrics_array_new[i, 6] = cv2.contourArea(conv_hull)
    
        m = cv2.moments(cont_coords)
        metrics_array_new[i, 4] = (m["m10"] / m["m00"]) if m["m00"] != 0.0 else 0.0
        metrics_array_new[i, 5] = (m["m01"] / m["m00"]) if m["m00"] != 0.0 else 0.0
        
        cont_coords_list.append((i, cont_coords))

        if not cont_coords_list:
            return [], np.empty((0, 5))
            
    idx = np.arange(total_conts, dtype=np.int16)
    metrics_array_new = np.column_stack([idx, metrics_array_new])
    logger.info(f"CONTOURS SIZE: {len(cont_coords_list)}")
    
    if metrics_array_new.size == 0:
        return [], np.empty((0, 9))
    
    has_child_all = (hierarchy[0, :, 2] != -1)
    has_child = has_child_all[idx]
    
    # Máscara reutilizable
    single_mask = bin_img.copy()
    single_mask[:] = 0
    # Variables para limpiar la región anterior
    prev_x, prev_y, prev_w, prev_h = np.zeros(4, np.uint8)
    for i, conts in cont_coords_list:

        if prev_w > 0 and prev_h > 0:
            single_mask[prev_y:prev_y+prev_h, prev_x:prev_x+prev_w] = 0
        
        x, y, w, h = bbox_cont[:, [0, 1, 2, 3]][i]
        hx = slice(y, y+h)
        wy = slice(x, x+w)
        
        # Dibuja el contorno actual
        cv2.drawContours(single_mask, [conts], -1, [255], cv2.FILLED)
        # Extrae solo la región de interés
        roi_mask = single_mask[hx, wy]
        roi_img = bin_img[hx, wy]
        pixels = roi_img[roi_mask == 255]
        pixels_outside = roi_img[roi_mask == 0]

        # Guarda el número de píxeles negros y totales
        metrics_array_new[i, 15] = np.count_nonzero(pixels == 255)  # COLUMNA 15
        
        # Verifica si HAY TINTA fuera del contorno (otro blob cerca)
        if np.count_nonzero(pixels_outside) > 0:
            metrics_array_new[i, 9] = 0         # COLUMNA 9
        else:
            metrics_array_new[i, 9] = 1         # COLUMNA 9

        if np.all(pixels==255):
            metrics_array_new[i, 8] = 1         # COLUMNA 8
        else:
            metrics_array_new[i, 8] = 0         # COLUMNA 8

        prev_x, prev_y, prev_w, prev_h = x, y, w, h

    bbox_cont = bbox_cont[:, 2] * bbox_cont[:, 3] # COLUMNA 14
    irregular_ratio = metrics_array_new[:, -2] / metrics_array_new[:, -1] # COLUMNA 16

    rec_widith = metrics_array_new[:, 2]
    rec_height = metrics_array_new[:, 3]
    angles = metrics_array_new[:, 4]
    convex_area = metrics_array_new[:, 7]
    # Para cada rectángulo, tomamos como min_side el lado (ancho/alto) más perpendicular/vertical al eje Y, es decir, el que corresponde al ángulo más cercano a 90°
    min_side = np.where(metrics_array_new[:, 4] > 45, rec_widith, rec_height)

    angle_norm = np.where(rec_widith < rec_height, angles + 90, angles)
    angle_norm = angle_norm % 180.0
    angles = angle_norm

    ratio_1 = convex_area / (rec_height * rec_widith)
    ratio_2 = (rec_height * rec_widith) / convex_area
    aspect_ratio = (np.maximum(rec_widith, rec_height) / np.minimum(rec_widith, rec_height))
    
    metrics_array_new = np.column_stack([
        metrics_array_new[:, 0],
        metrics_array_new[:, 1],
        metrics_array_new[:, 2],
        metrics_array_new[:, 3],
        metrics_array_new[:, 4],
        metrics_array_new[:, 5],
        metrics_array_new[:, 6],
        metrics_array_new[:, 7],
        metrics_array_new[:, 8],
        metrics_array_new[:, 9],  
        aspect_ratio,                   # COLUMNA 10
        has_child,                      # COLUMNA 11
        ratio_1,                        # COLUMNA 12
        ratio_2,                        # COLUMNA 13
        bbox_cont,                      # COLUMNA 14
        metrics_array_new[:, 15], 
        irregular_ratio,                # COLUMNA 16
        min_side                        # COLUMNA 17
    ])

    metrics_array_new = metrics_array_new[[c[0] for c in cont_coords_list]]
    valid_coords = cont_coords_list

    valid_contours = len(valid_coords)
    matrix_size = metrics_array_new.shape[0]
    if valid_contours != matrix_size:
        logger.warning(f"Contornos dispares: {valid_contours} != {matrix_size}")
        return [], np.empty((0, 5))
    centroids = metrics_array_new[:, 5:7]
    variances = np.array([np.var(np.linalg.norm(coords.reshape(-1, 2) - c, axis=1)) for (_, coords), c in zip(valid_coords, centroids)], dtype=np.float32)
    metrics_array_new = np.column_stack([metrics_array_new, variances])

    # logger.info(f"METRICS LIST: {metrics_array.shape}")
    logger.info(f"Tiempo extrayendo métricas VECTOROIZADAS: {time.perf_counter()-time0:.6f}'s")
    return valid_coords, metrics_array_new

def is_binarized(img: np.ndarray[Any, Any]) -> bool:
    """Devuelve True si la imagen solo contiene dos valores únicos (0 y 255)."""
    if img.dtype != np.uint8:
        return False
    unique_vals = np.unique(img)
    return unique_vals.size == 2 and set(unique_vals) <= {0, 255}