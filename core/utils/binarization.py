# PerfectOCR/core/utils/binarization.py
import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from skimage.filters import threshold_sauvola  # type: ignore

logger = logging.getLogger(__name__)

def binarice(cropped_img: np.ndarray[Any, np.dtype[np.uint8]], worker_config: Dict[str, Any], min_area: int) -> Dict[str, Any]:
    """Binariza, mide blobs (componentes conectados) y guarda sus métricas"""
    c_value: int = worker_config.get('c_value', 7)
    height_thresholds: List[int] = worker_config['height_thresholds_px']
    block_sizes_map: List[int] = worker_config['block_sizes_map']
    min_blobs_for_frag: int = worker_config.get('min_blobs_for_frag', 2)
    gap_threshold_norm: float = worker_config.get('gap_threshold_norm', 0.05)
    # min_area_factor =  worker_config.get('min_area_factor', 0.005) 
    
    height = int(cropped_img.shape[0])
    # widith = int(cropped_img.shape[1])
    # area = widith * height
    # min_area = area * min_area_factor
    
    block = _get_adaptive_block_size(height, height_thresholds, block_sizes_map)
    mode: str = _measure_polygon_quality(cropped_img)

    if mode == "otsu":
        bin_img = _otsu_binarize(cropped_img)
    elif mode == "adaptive_gaussian":
        bin_img = _adaptive_binarize(cropped_img, block, c_value)
    elif mode == "sauvola":
        bin_img = _sauvola_binarize(cropped_img, block)
    else:
        bin_img = _adaptive_mean_fallback(cropped_img, block, c_value)

    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logger.info("No contornos")
        
    valid_contours = [c for c in contours if cv2.contourArea(c) < min_area]
    valid_boxes_norm: List[List[float]] = []
    blob_metrics: Dict[str, Any] = {}
    
    for c in contours:

        if not valid_contours:
            continue
        
        x, y, w, h = cv2.boundingRect(c)
        
        x2, y2 = x + w, y + h
        valid_boxes_norm.append([
            x / bin_img.shape[1], y / bin_img.shape[0],
            x2 / bin_img.shape[1], y2 / bin_img.shape[0]
        ])

    if valid_boxes_norm:
        sorted_boxes = sorted(valid_boxes_norm, key=lambda box: box[0])
        num_blobs = len(contours)
    
        gaps_x_norm: List[float] = []
        if num_blobs > 1:
            for i in range(num_blobs - 1):
                gap = sorted_boxes[i+1][0] - sorted_boxes[i][2]
                gaps_x_norm.append(max(00, float(gap)))

        needs_fragmentation = (num_blobs >= min_blobs_for_frag and (max(gaps_x_norm) if gaps_x_norm else 00) >= float(gap_threshold_norm))

        if needs_fragmentation:
            blob_metrics = {
                "needs_fragmentation": needs_fragmentation,
                "num_blobs": num_blobs,
                "blobs_norm_boxes": sorted_boxes,
                "gaps_x_norm": gaps_x_norm
                }

            # logger.info(f"blob_metrics: {blob_metrics}")

    return blob_metrics
    
def _get_adaptive_block_size(height: float, height_thresholds: List[int], block_sizes_map: List[int]) -> int:
    for i, threshold in enumerate(height_thresholds):
        if height <= threshold:
            block_size = block_sizes_map[min(i, len(block_sizes_map) - 1)]
            return max(3, block_size if block_size % 2 != 0 else block_size + 1)
    final_block_size = block_sizes_map[-1]
    return max(3, final_block_size if final_block_size % 2 != 0 else final_block_size + 1)

def _measure_polygon_quality(cropped_img: np.ndarray[Any, np.dtype[np.uint8]]) -> str:
    std = np.std(cropped_img)
    if std == 0: return "adaptive_mean"        
    hist = cv2.calcHist([cropped_img], [0], None, [256], [0, 256]).flatten()
    peaks = np.sum((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:]))
    prob = hist / np.sum(hist)
    entropy = -np.sum(prob * np.log2(prob + 1e-8))

    if peaks >= 2 and std > 30: 
        return "otsu"
        
    elif std > 20 and entropy > 45: 
        return "adaptive_gaussian"
        
    elif std > 10: 
        return "sauvola"
        
    else: 
        return "adaptive_mean"
    
def _otsu_binarize(cropped_img: np.ndarray[Any, np.dtype[np.uint8]]) -> np.ndarray[np.int8, Any]:
    _, bin_img = cv2.threshold(cropped_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bin_img

def _adaptive_binarize(cropped_img: np.ndarray[Any, np.dtype[np.uint8]], block_size: int, c_value: int) -> np.ndarray[np.int8, Any]:
    return cv2.adaptiveThreshold(
        cropped_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, block_size, c_value
    )

def _sauvola_binarize(cropped_img: np.ndarray[Any, np.dtype[np.uint8]], adaptive_block_size: int) -> np.ndarray[np.int8, Any]:
    """Sauvola thresholding producing uint8 mask with text as foreground (0) and background (255)"""
    thresh_sauvola = threshold_sauvola(cropped_img, window_size=adaptive_block_size) # type: ignore
    bin_bool: np.ndarray[np.uint8, Any] = (cropped_img > thresh_sauvola)# type: ignore
    bin_img = (bin_bool.astype(np.uint8) * 255)# type: ignore
    return bin_img# type: ignore

def _adaptive_mean_fallback(cropped_img: np.ndarray[Any, np.dtype[np.uint8]], block_size: int, c_value: int) -> np.ndarray[np.int8, Any]:
    return cv2.adaptiveThreshold(
        cropped_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, block_size, max(1, c_value - 2) 
    )

def visual_analysis_needs_fragmentation(bin_img: np.ndarray[np.int8, Any], min_area: int, min_blobs_for_frag: int) -> Tuple[bool, Any]:
    """Determina si la imagen binarizada parece contener múltiples elementos separados
    Devuelve un bool indicando necesidad de fragmentar y la lista de contornos válidos"""        
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, []

    # Filtrar contornos muy pequeños que son probablemente ruido
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    return (len(valid_contours) >= min_blobs_for_frag), valid_contours
