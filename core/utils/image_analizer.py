# PerfectOCR/core/utils/image_analizer.py
import cv2
import numpy as np
import logging
from typing import Dict, Any, List
from core.utils.image_utils import binarice_img
from core.utils.text_encoder import text_compacter

logger = logging.getLogger(__name__)

def analize_bin_img(img: np.ndarray[Any, Any], worker_config: Dict[str, Any], binarice: bool) -> Dict[str, Any]:

    min_cc_for_frag: int = worker_config.get("min_cc_for_frag", 2)
    text: str = worker_config.get("text", "")
    if (len(text.split()) - 1) < min_cc_for_frag:
        # logger.info(f"Unico texto, no se analizará")
        return {}
    
    if binarice:
        bin_img = binarice_img(img, worker_config)
    else:
        bin_img = img    

    cc_metrics = extract_cc_metrics(bin_img, worker_config, text)

    if not cc_metrics or len(cc_metrics["cc"]) == 0:
        logger.debug("No se detectaron componentes conectados válidos tras el filtrado.")
        return {"needs_fragmentation": False,
                "cause": "not_cc_metrics"
                }

    needs_fragmentation = decide_split(cc_metrics, worker_config)

    if not needs_fragmentation:
        return {
            "needs_fragmentation": needs_fragmentation,
            "cause": "not_enought_data"
        }
    num_blobs = cc_metrics["num_blobs"]
    blob_metrics: Dict[str, Any] = {
        "needs_fragmentation": needs_fragmentation,
        "num_blobs": num_blobs,
    }

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

def extract_cc_metrics(bin_img: np.ndarray[Any, np.dtype[np.uint8]], worker_config: Dict[str, Any], text: str) -> Dict[str, Any]:
    """
    Calcula métricas de CC robustas, filtrando ruido (rayones, manchas)
    usando Área y Solidez.
    bin_img: np.uint8, foreground=255, background=0
    """
    percentile = worker_config["percentile"]
    num_words = len(text.split())
    wgaps = num_words - 1 
    chars = list(text_compacter(text))
    clean_txt: List[str] = []
    for ch in chars: 
        if ch.isalnum() and not ch.isspace():
            clean_txt.append(ch)
        else:
            chars.remove(ch)

    max_elements = len(clean_txt)

    # logger.info(f"Origial: {text}, limpio: {clean_txt}, gaps: {wgaps}")

    connectivity = worker_config.get("connectivity", {})
    min_area_factor = worker_config.get("min_area_factor", {})
    solidity_threshold =  worker_config.get("solidity_threshold", {})
    bbox_area =  bin_img.size
    min_area = bbox_area*min_area_factor

    # 1. Etiquetado rápido
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img, connectivity)

    #logger.info(f"numero de labels: {n_labels}")

    areas = stats[1:, cv2.CC_STAT_AREA]  # Excluye el fondo (label 0)
    sorted_indices = np.argsort(areas)[::-1]  # Índices ordenados (0 es el blob más grande)
    top_labels = sorted_indices[:max_elements] + 1  # +1 porque stats[0] es fondo
    areas_array = np.array(areas, dtype=np.float32)
    min_area_per=np.percentile(areas_array, percentile)[1]
   # logger.info(f"Factor: {min_area}, Percentil: {min_area_per}")

    cc_list: List[Any]= []

    for top_labels in range(1, n_labels):
        x, y, w, h, area = (stats[top_labels, cv2.CC_STAT_LEFT], stats[top_labels, cv2.CC_STAT_TOP], stats[top_labels, cv2.CC_STAT_WIDTH], stats[top_labels, cv2.CC_STAT_HEIGHT], stats[top_labels, cv2.CC_STAT_AREA])

        # cx, cy = centroids[lab]
        # logger.info(f"Componente {lab}: x={x}, y={y}, w={w}, h={h}, area={area}, centro=({cx},{cy})")
        # logger.info(f"{lab}, {n_labels}")
        if area < min_area:
            continue
        
        # logger.info(f"Area aprobada: {area}")
        # --- Filtro 3: Solidez (Filtro robusto contra manchas/ruido) ---
        try:
            # Crear una máscara solo para este blob
            component_mask: np.ndarray[Any, np.dtype[np.uint8]] = ((labels == top_labels).astype(np.uint8))
            
            # Encontrar contornos solo en esta máscara (muy rápido)
            contours = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
            logger.info(f"{len(contours)} vs {top_labels}")
        #    logger.info(f"{np.mean(component_mask)}")

            if not contours:
                logger.warning("Sin contornos")
                return {}
            
            c = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(c)
            hull_area: float = cv2.contourArea(hull)
            # logger.info(f"HULL: {hull}, HULL AREA: {hull_area}")

            if hull_area == 0.0:
                continue

            solidity = float(area) / hull_area

            logger.info(f"SOLIDUTY:{solidity}")

            # Descartar formas "dispersas" o "huecas" (ruido)
            if solidity < solidity_threshold:
                continue

        except cv2.error:
            continue  # Proteger contra contornos degenerados

        cx, cy = centroids[top_labels]
        cc_list.append({
            "label": top_labels,
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "area": int(area),
            "cx": float(cx), "cy": float(cy)
        })

    if not cc_list:
        # logger.error("SIN CCLIST")
        return {}

    heights = np.array([c["h"] for c in cc_list], dtype=np.float32)
    H_median = np.median(heights).astype(np.float32)
    total_area_cc = float(sum([c["area"] for c in cc_list]))
    density = float(total_area_cc) / float(bbox_area)

    # Normalizar características por H_mean (inmune a DPI)
    for c in cc_list:
        c["w_norm"] = c["w"] / max(1.0, H_median)
        c["h_norm"] = c["h"] / max(1.0, H_median)
        c["area_norm"] = c["area"] / max(1.0, bbox_area)

    # Ordenar por 'x' (para calcular gaps)
    cc_sorted: List[Any] = sorted(cc_list, key=lambda z: z["x"])

    gaps: List[Any] = []
    for i in range(len(cc_sorted) - 1):
        gap_px = cc_sorted[i + 1]["x"] - (cc_sorted[i]["x"] + cc_sorted[i]["w"])
        # Normalizar gap por la altura media de caracteres
        gaps.append(max(0.0, gap_px) / max(1.0, H_median))

    # logger.info(f"DIVISIONES PRECALCULADAS: '{num_words}', NUM_BLOBS: '{len(cc_sorted)}")
    metrics: Dict[str, Any] = {
        "cc": cc_sorted,
        "H_median": H_median,
        "density": density,
        "gaps_norm": gaps,
        "num_blobs": len(cc_sorted),
    }
    return metrics

def decide_split(metrics: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    """
    Toma las métricas de CC robustas y aplica reglas heurísticas
    para decidir si un polígono debe ser fragmentado.
    """
    k_sigma = cfg.get("k_sigma", {})
    min_cc_for_frag: int = cfg.get("min_cc_for_frag", 2)
    min_gap_outlier = cfg.get("min_gap_outlier", {})
    density_threshold = cfg.get("density_threshold", {})
    width_threshold = cfg.get("width_var_threshold", {})
    n_cc = len(metrics["cc"])

    # Regla 1: No fragmentar si hay muy pocos blobs
    if n_cc < min_cc_for_frag:
        # logger.info(f"Min contonrnos: {n_cc} < {min_cc_for_frag}")
        return False

    # Regla 2: No fragmentar si es muy denso (probablemente una palabra)
    if metrics["density"] > density_threshold:
        logger.info(f"Densidad: {metrics["density"]} > {density_threshold}")
        return False

    # Regla 3: No fragmentar si todos los caracteres son de ancho similar
    w_norms = np.array([c["w_norm"] for c in metrics["cc"]])
    if np.std(w_norms) < width_threshold:
        # logger.info(f"Vairanza de largo: {np.var(w_norms)} < {width_threshold}")
        return False

    # Regla 4: Fragmentar si hay gaps atípicos (grandes)
    gaps = np.array(metrics["gaps_norm"]) if metrics["gaps_norm"] else np.array([0.0])

    if len(gaps) == 0:
        return False

    mu, sigma = gaps.mean(), gaps.std()

    # Buscar gaps que sean (A) atípicos (k*sigma) Y (B) suficientemente grandes (min_gap)
    threshold = max(mu + k_sigma * max(1e-6, sigma), min_gap_outlier)
    split_indices = np.where(gaps > threshold)[0]

    if len(split_indices) > 0:
        # logger.info(f"Split indices insuficientes")
        return True

    return False
