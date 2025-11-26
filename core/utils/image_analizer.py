# PerfectOCR/core/utils/image_analizer.py
import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from services.output_service import save_croped_image
from core.utils.image_utils import binarice_img
from core.utils.text_encoder import text_compacter
from core.utils.text_validator import valid_punt_chars

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

    cc_metrics, _, _ = analice_cc_metrics(bin_img, worker_config)

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

def extract_cc_metrics(img: np.ndarray[Any, np.dtype[np.uint8]], worker_config: Dict[str, Any], binarice: Optional[bool] = False) -> Tuple[Dict[int, Any], Optional[np.ndarray[Any, np.dtype[np.uint8]]]]:
    """
    Calcula métricas de CC robustas, filtrando ruido (rayones, manchas)
    usando Área y Solidez.
    bin_img: np.uint8, foreground=255, background=0
    """
    if not binarice or binarice is None:
        binarice = False
        bin_img = img

    else:
        bin_img = binarice_img(img, worker_config={})
  
    #poly = worker_config["poly_id"]

    connectivity: int = worker_config.get("connectivity", 8)
    
    # 1. Etiquetado rápido
    #logger.info(f"{poly}")
   
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logger.info(f"Numero de contornos: {len(contours)}")
    cont_array_dict: Dict[int, Dict[str, np.ndarray[Any, np.dtype[np.uint16]]] | float]= {}
    
    for i, cont in enumerate(contours):
        cont_coords = cont.reshape(-1, 2).astype(np.int32)
        cont_bbox = cv2.boundingRect(cont_coords)
        convex_hull = np.array(cv2.convexHull(cont_coords))
        cont_area = cv2.contourArea(cont_coords)
        hull_area = cv2.contourArea(convex_hull)

        logger.info(f"Promedio de contorno '{i}': {cv2.mean(cont_coords)}")
        
        cont_array_dict[i] = {
            "cont_coords": cont_coords,
            "cont_bbox": cont_bbox,
            "convex_hull": convex_hull,
            "cont_area": cont_area,
            "hull_area": hull_area
        }
    
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img, connectivity= connectivity)

    label_labeled = np.arange(1, n_labels).astype(np.uint16)
    cx, cy = np.hsplit(centroids[:-1], (2))    
    mapped_stats = np.column_stack([label_labeled, stats[1:, cv2.CC_STAT_AREA], stats[1:, cv2.CC_STAT_LEFT], stats[1:, cv2.CC_STAT_TOP], stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT], cx, cy]).astype(np.uint16)
   # logger.info(f"MAPPED_STATS: {mapped_stats.shape}") 
    logger.debug(f"LABELS: {labels[1:].shape}, CENTROIDS: {centroids[:-1].shape}, N LABLES: {n_labels-1}")
  #  logger.info(f"Tmaño imagne: '{bin_img.shape}'")
   # unique_labs = np.unique(labels, return_index=True, axis=0)[0]

    # logger.info(f"LABELS: {unique_labs}")
   ##extract_labs = np.extract(unique_labs, labels)
   # logger.info(f"{extract_labs}")

    image_metrics: Dict[str, Any] = { 
        "mapped_stats": mapped_stats, 
        "cont_array_dict": cont_array_dict
    }
    if binarice:
        return image_metrics, bin_img
    
    else:
        return image_metrics, None

def analice_cc_metrics(bin_img: np.ndarray[Any, np.dtype[np.uint8]], worker_config: Dict[str, Any]) -> Dict[str, Any]:
   
    text = worker_config.get("text", "")
    valid_punt = valid_punt_chars()
    poly = worker_config.get("poly_id")
    solidity_threshold = worker_config.get("solidity_threshold")
    # percentile: Tuple[float, float] = worker_config["percentile"]
    # num_words = len(text.split())
    # wgaps = num_words - 1
    chars = list(text_compacter(text))
    clean_txt: List[str] = []
    for ch in chars:
        if not ch.isalnum():
            if ch in valid_punt:
                continue 

            if not ch.isascii():
                continue 

            clean_txt.append(ch)
        clean_txt.append(ch)
   
    text_array = np.array(clean_txt, dtype=np.unicode_)
    mapped_stats, cont_array_dict = extract_cc_metrics(bin_img, worker_config)

    sorted_cc = np.sort(mapped_stats[:, 1])[::-1]

    sorted_labels = sorted_cc[:len(text_array)]
    
    condition = np.isin(mapped_stats[: ,1], sorted_labels, invert=False)
    top_labels = np.compress(condition, mapped_stats, axis=0)
    
    order = np.argsort(top_labels[:, 2]).astype(np.uint16)
    reordered = top_labels[order]
    full_array = np.column_stack([reordered, text_array])

    # logger.info(f"FULL ARRAY:"
    #             "\n"f"{full_array}, SHAPE: {full_array.shape}") 
    # logger.info(f"{full_array[0]}")
    
        # Recorre cada blob y guarda su recorte
    cc_list: List[Any] = []
    for i in range(full_array.shape[0]):
        pos = int(full_array[i, 0])
        area = int(full_array[i, 1])
        x = int(full_array[i, 2])
        y = int(full_array[i, 3])
        w = int(full_array[i, 4])
        h = int(full_array[i, 5])
        cx = int(full_array[i, 6])
        cy = int(full_array[i, 7])
        
        try:
            # if area < min_area:
            #     # logger.info(f"Demasiado pequeño")
            #     continue
            # Crear una máscara solo para este blob
            component_mask: np.ndarray[Any, np.dtype[np.uint8]] = (labels[:, 1]).astype(np.uint8)
            
            # Encontrar contornos solo en esta máscara (muy rápido)
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                logger.warning("Sin contornos")
                return {}

            c = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(c)
            hull_area: float = cv2.contourArea(hull)

            if hull_area == 0.0:
                # logger.info(f"{poly} descartado por area de hull")
                continue

            solidity = float(area) / hull_area

            # Descartar formas "dispersas" o "huecas" (ruido)
            if solidity < solidity_threshold:
                # logger.info(f"{poly} descartada por solidez")
                continue

        except cv2.error:
            continue  # Proteger contra contornos degenerados

        cc_list.append({
            "label": pos,
            "contours": contours,
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "area": int(area),
            "cx": float(cx), "cy": float(cy)
        })

    if not cc_list:
        # logger.error("SIN CCLIST")
        return {}
   
    bbox_area = bin_img.size

    heights = np.array([c["h"] for c in cc_list], dtype=np.float32)
    H_median = np.median(heights).astype(np.float32)
    total_area_cc = float(sum([c["area"] for c in cc_list]))
    density = float(total_area_cc) / float(bbox_area)

    # Normalizar características por H_mean (inmune a DPI)
    for c in cc_list:
        logger.info(f"{c["label"]}")
        c["w_norm"] = c["w"] / max(1.0, H_median)
        c["h_norm"] = c["h"] / max(1.0, H_median)
        c["area_norm"] = c["area"] / max(1.0, bbox_area)

    # Ordenar por 'x' (para calcular gaps)
    cc_sorted: List[Any] = sorted(cc_list, key=lambda z: z["x"])

    logger.info(f"Cantidad de cc:{len(cc_sorted)}")

    output = worker_config.get("output")
    if output:
        worker_name = "image_analizer"
        image_name = worker_config["image_name"]
        output_paths = worker_config["output_paths"]
        
        for idx, c in enumerate(cc_sorted):
            poly_id = f"{poly}_{c['label']}_{idx}"
            y, x = c["y"], c["x"]
            h, w = c["h"], c["w"]
            
            # Recortar usando coordenadas de píxeles, NO normalizadas
            cropped_img = bin_img[y:y+h, x:x+w]
            
            save_croped_image(
                image_name, poly_id, cropped_img, output_paths,
                worker_name, method="components"
            )

    gaps: List[Any] = []
    for i in range(len(cc_sorted) - 1):
        gap_px = cc_sorted[i + 1]["x"] - (cc_sorted[i]["x"] + cc_sorted[i]["w"])
        # Normalizar gap por la altura media de caracteres
        gaps.append(max(0.0, gap_px) / max(1.0, H_median))

    # logger.info(f"DIVISIONES PRECALCULADAS: '{num_words}', NUM_BLOBS: '{len(cc_sorted)}")
    metrics: Dict[str, Any] = {
        #"contours": contours,
        "cc": cc_sorted,
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
