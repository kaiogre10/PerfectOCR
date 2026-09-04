# PerfectOCR/core/utils/image_utils.py
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
            
        if is_binarized(img):
            return make_contiguous(img)

        try:
            img_arr = np.asarray(img, dtype=np.uint8, order="C")
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

def validate_image(img: Optional[np.ndarray[Any, Any]]) -> bool:
    return bool(7 < int(np.mean(img)) < 251) if img is not None else False

def use_bilateral_filter(img: np.ndarray[Any, np.dtype[np.uint8]], d: int, sigma_color: int, sigma_space: int)-> np.ndarray[Any, np.dtype[np.uint8]]:
    return make_contiguous(cv2.bilateralFilter(img, d, sigma_color, sigma_space))

def use_sobel(img: np.ndarray[Any, np.dtype[np.uint8]], ksize: int) -> float:
    return float(np.mean(np.abs(cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=ksize)), dtype=np.float32))

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
    """Sauvola thresholding producing uint8 mask with text as foreground (0) and background (255)"""
    thresh_sauvola = threshold_sauvola(image=cropped_img, window_size=adaptive_block_size) 
    bin_bool = (cropped_img > thresh_sauvola)
    return make_contiguous(bin_bool * 255)

def adaptive_mean_fallback(cropped_img: np.ndarray[Any, np.dtype[np.uint8]], block_size: int, c_value: int) -> np.ndarray[Any, np.dtype[np.uint8]]:
    return make_contiguous(cv2.adaptiveThreshold(cropped_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, max(1, c_value - 2)))

def decolorate(full_img: np.ndarray[Any, Any]) -> np.ndarray[Any, np.dtype[np.uint8]]:
    """Elimina colores (rayones, resaltados, etc.) de la imagen, dejando solo blanco y negro."""
    if is_binarized(full_img):
        logger.error(f"IMAGEN BINARIA EN EL INICIO")
        return make_contiguous(full_img) 
        
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
    if gray is None or not validate_image(gray):
        logger.warning("Normalice IMG devolvío imagen, Imagen en grises de cv2")
        return make_contiguous(cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY))
    else:
        return make_contiguous(gray)

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

def get_contours_values(img: np.ndarray[Any, np.dtype[np.uint8]], binary: Optional[bool] = False) -> Tuple[List[Tuple[int, np.ndarray[Any, np.dtype[np.int32]]]], np.ndarray[Any, Any]]:
    """Calcula UNICAMENTE los features de OPEN CV"""
    time0 = time.perf_counter()
    if binary or is_binarized(img):
        logger.info("Imagen ya binaria")
        bin_img = img
    else:
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