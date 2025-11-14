# PerfectOCR/core/utils/binarizator.py
import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from skimage.filters import threshold_sauvola  # type: ignore

logger = logging.getLogger(__name__)

def binarice_img(cropped_img: np.ndarray[Any, np.dtype[np.uint8]], worker_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Binariza la imagen, extrae métricas robustas de componentes conectados (CC) y decide si necesita fragmentación.
    """
    c_value: int = worker_config.get('c_value', {})
    height_thresholds: List[int] = worker_config['height_thresholds_px']
    block_sizes_map: List[int] = worker_config['block_sizes_map']
    min_area_factor: float = worker_config.get('min_area_factor', {})
    height = int(cropped_img.shape[0])
    area = cropped_img.size

    area_min = area*min_area_factor
    block = get_adaptive_block_size(height, height_thresholds, block_sizes_map)
    mode: str = measure_polygon_quality(cropped_img)

    if mode == "otsu":
        bin_img = otsu_binarize(cropped_img)
    elif mode == "adaptive_gaussian":
        bin_img = adaptive_binarize(cropped_img, block, c_value)
    elif mode == "sauvola":
        bin_img = sauvola_binarize(cropped_img, block)
    else:
        bin_img = adaptive_mean_fallback(cropped_img, block, c_value)

    bin_img = cv2.bitwise_not(bin_img).astype(np.uint8)
    
    cc_metrics = extract_cc_metrics(bin_img, area_min)

    if not cc_metrics or cc_metrics["n_cc"] == 0:
        logger.debug("No se detectaron componentes conectados válidos tras el filtrado.")
        return {}

    needs_fragmentation, reason_info = decide_split(cc_metrics, worker_config)

    blob_metrics: Dict[str, Any] = {
        "needs_fragmentation": needs_fragmentation,
        "fragmentation_reason_info": reason_info,
        "num_blobs": cc_metrics["n_cc"],
        "density": cc_metrics["density"],
        "mean_height": cc_metrics["H_mean"],
        "gaps_norm": cc_metrics["gaps_norm"],
    }

    # Añadir los bounding boxes normalizados si existen
    if cc_metrics["cc"]:
        img_h, img_w = bin_img.shape[:2]
        valid_boxes_norm: List[List[int]] = []
        for c in cc_metrics["cc"]:
            x, y, w, h = c["x"], c["y"], c["w"], c["h"]
            x2, y2 = x + w, y + h
            valid_boxes_norm.append([
                x / img_w, y / img_h,
                x2 / img_w, y2 / img_h
            ])
        # Los boxes ya están ordenados por 'x' desde extract_cc_metrics
        blob_metrics["blobs_norm_boxes"] = valid_boxes_norm

    # logger.info(f"Metricas: {blob_metrics}")
    return blob_metrics

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
    hist = cv2.calcHist([cropped_img], [0], None, [256], [0, 256]).flatten()
    peaks = np.sum((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:]))
    prob = hist / np.sum(hist)
    entropy = -np.sum(prob * np.log2(prob + 1e-8))

    if peaks >= 2 and std > 30:
        return "otsu"  # Alto contraste, bimodal

    elif std > 20 and entropy > 5.0:
        return "adaptive_gaussian"  # Contraste variable

    elif std > 10:
        return "sauvola"  # Texto sobre fondo no uniforme

    else:
        return "adaptive_mean"  # Bajo contraste, imagen "plana"

def otsu_binarize(cropped_img: np.ndarray[Any, np.dtype[np.uint8]]) -> np.ndarray[Any, np.dtype[np.uint8]]:
    resultis = cv2.threshold(cropped_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_img = resultis[1].astype(np.uint8)
    return bin_img

def adaptive_binarize(cropped_img: np.ndarray[Any, np.dtype[np.uint8]], block_size: int, c_value: int) -> np.ndarray[Any, np.dtype[np.uint8]]:
    return cv2.adaptiveThreshold(cropped_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c_value).astype(np.uint8)

def sauvola_binarize(cropped_img: np.ndarray[Any, np.dtype[np.uint8]], adaptive_block_size: int) -> np.ndarray[Any, np.dtype[np.uint8]]:
    """Sauvola thresholding producing uint8 mask with text as foreground (0) and background (255)"""
    thresh_sauvola = threshold_sauvola(cropped_img, window_size=adaptive_block_size)  # type: ignore
    bin_bool: np.ndarray[np.uint8, Any] = (cropped_img > thresh_sauvola)  # type: ignore
    bin_img = (bin_bool.astype(np.uint8) * 255)  # type: ignore
    return bin_img  # type: ignore

def adaptive_mean_fallback(cropped_img: np.ndarray[Any, np.dtype[np.uint8]], block_size: int, c_value: int) -> np.ndarray[Any, np.dtype[np.uint8]]:
    return cv2.adaptiveThreshold(
        cropped_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, max(1, c_value - 2)).astype(np.uint8)

def extract_cc_metrics(bin_img: np.ndarray[Any, np.dtype[np.uint8]], min_area: float) -> Dict[str, Any]:
    """
    Calcula métricas de CC robustas, filtrando ruido (rayones, manchas)
    usando Área, Ratio de Aspecto y Solidez.
    """
    # bin_img: np.uint8, foreground=255, background=0
    # 1. Etiquetado rápido
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img, connectivity=8)

    bbox_area = bin_img.shape[0] * bin_img.shape[1]
    if bbox_area == 0: return {}

    cc_list = []

    for lab in range(1, n_labels):
        x, y, w, h, area = (
            stats[lab, cv2.CC_STAT_LEFT], stats[lab, cv2.CC_STAT_TOP],
            stats[lab, cv2.CC_STAT_WIDTH], stats[lab, cv2.CC_STAT_HEIGHT],
            stats[lab, cv2.CC_STAT_AREA]
        )

        # --- Filtro 1: Área Mínima (Filtro rápido) ---
        if area < min_area:
            continue

        # --- Filtro 3: Solidez (Filtro robusto contra manchas/ruido) ---
        try:
            # Crear una máscara solo para este blob
            component_mask: np.ndarray[Any, np.dtype[np.uint8]] = ((labels == lab).astype(np.uint8) * 255)
            # Encontrar contornos solo en esta máscara (muy rápido)
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                continue

            c = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(c)
            hull_area: float = cv2.contourArea(hull)

            if hull_area == 0.0:
                continue

            solidity = float(area) / hull_area

            # Descartar formas "dispersas" o "huecas" (ruido)
            if solidity < 0.4:
                continue
        except cv2.error:
            continue  # Proteger contra contornos degenerados

        # --- Si pasa todos los filtros, es un blob válido ---
        cx, cy = centroids[lab]
        cc_list.append({
            "label": lab,
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "area": int(area),
            "cx": float(cx), "cy": float(cy)
        })

    if not cc_list:
        return {"cc": [], "H_mean": 0, "density": 0, "gaps_norm": [], "n_cc": 0}

    heights = np.array([c["h"] for c in cc_list], dtype=np.float32)
    H_mean: float = heights.mean()
    total_area_cc = float(sum([c["area"] for c in cc_list]))
    density = float(total_area_cc) / float(bbox_area)

    # Normalizar características por H_mean (inmune a DPI)
    for c in cc_list:
        c["w_norm"] = c["w"] / max(1.0, H_mean)
        c["h_norm"] = c["h"] / max(1.0, H_mean)
        c["area_norm"] = c["area"] / max(1.0, bbox_area)

    # Ordenar por 'x' (para calcular gaps)
    cc_sorted: List[float] = sorted(cc_list, key=lambda z: z["x"])

    gaps: List[int] = []
    for i in range(len(cc_sorted) - 1):
        gap_px = cc_sorted[i + 1]["x"] - (cc_sorted[i]["x"] + cc_sorted[i]["w"])
        # Normalizar gap por la altura media de caracteres
        gaps.append(max(0.0, gap_px) / max(1.0, H_mean))

    metrics: Dict[str, Any] = {
        "cc": cc_sorted,
        "H_mean": H_mean,
        "density": density,
        "gaps_norm": gaps,
        "n_cc": len(cc_sorted)
    }
    return metrics

def decide_split(metrics: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    Toma las métricas de CC robustas y aplica reglas heurísticas
    para decidir si un polígono debe ser fragmentado.
    """
    k_sigma = cfg.get("k_sigma", {})
    min_cc_for_frag: int = cfg.get("min_cc_for_frag", {})
    min_gap_outlier = cfg.get("min_gap_outlier", {})
    density_threshold = cfg.get("density_threshold", {})
    max_cc_for_density_rule = cfg.get("max_cc_for_density_rule", {})
    width_var_threshold = cfg.get("width_var_threshold", {})
    n_cc = metrics["n_cc"]

    # Regla 1: No fragmentar si hay muy pocos blobs
    if n_cc < min_cc_for_frag:
        return False, {}

    # Regla 2: No fragmentar si es muy denso (probablemente una palabra)
    if metrics["density"] > density_threshold and n_cc <= max_cc_for_density_rule:
        return False, {}

    # Regla 3: No fragmentar si todos los caracteres son de ancho similar
    w_norms = np.array([c["w_norm"] for c in metrics["cc"]])
    if np.var(w_norms) < width_var_threshold:
        return False, {}

    # Regla 4: Fragmentar si hay gaps atípicos (grandes)
    gaps = np.array(metrics["gaps_norm"]) if metrics["gaps_norm"] else np.array([0.0])

    if len(gaps) == 0:
        return False, {}

    mu, sigma = gaps.mean(), gaps.std()

    # Buscar gaps que sean (A) atípicos (k*sigma) Y (B) suficientemente grandes (min_gap)
    threshold = max(mu + k_sigma * max(1e-6, sigma), min_gap_outlier)
    split_indices = np.where(gaps > threshold)[0]

    if len(split_indices) > 0:
        reason_info: Dict[str, Any] = {
            "reason": "gap_outlier",
            "positions_idx": split_indices.tolist(),
            "gaps_found": gaps[split_indices].tolist(),
            "threshold": threshold,
            "mu": float(mu),
            "sigma": float(sigma)
        }
        return True, reason_info

    return False, {}
