# PerfectOCR/core/utils/binarization.py
import cv2
import numpy as np
import logging
from typing import Dict, Any, List
from skimage.filters import threshold_sauvola  # type: ignore
from skimage.measure import regionprops, label

logger = logging.getLogger(__name__)

def binarice(cropped_img: np.ndarray[Any, np.dtype[np.uint8]], worker_config: Dict[str, Any]) -> Dict[str, Any]:
    """Binariza, mide blobs (componentes conectados) y guarda sus métricas"""
    c_value: int = worker_config.get('c_value', {})
    height_thresholds: List[int] = worker_config['height_thresholds_px']
    block_sizes_map: List[int] = worker_config['block_sizes_map']
    min_blobs_for_frag: int = worker_config.get('min_blobs_for_frag', {})
    gap_threshold_norm: float = worker_config.get('gap_threshold_norm', {})
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

    # 1. Aplicar la corrección #1 (inversión) aquí arriba
    bin_img = cv2.bitwise_not(bin_img)
    
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) > area_min]
    (bin_img)
    # 2. Aplicar la corrección #2 (filtro > min_area)
    valid_boxes_norm: List[List[float]] = []
    blob_metrics: Dict[str, Any] = {}
    # logger.info(f"contours: {valid_contours}")
    # 3. Iterar sobre los contornos VÁLIDOS
    for c in valid_contours:
        x, y, w, h = cv2.boundingRect(c)
        x2, y2 = x + w, y + h
        valid_boxes_norm.append([
            x / bin_img.shape[1], y / bin_img.shape[0],
            x2 / bin_img.shape[1], y2 / bin_img.shape[0]
        ])

    # 4. Ahora todas las métricas están sincronizadas
    sorted_boxes = sorted(valid_boxes_norm, key=lambda box: box[0])
    num_blobs = len(valid_contours)
    
    gaps_x_norm: List[float] = []
    if num_blobs > 1:
        for i in range(num_blobs - 1):
            gap = sorted_boxes[i+1][0] - sorted_boxes[i][2]
            # Corregido '00' a '0.0' para claridad
            gaps_x_norm.append(max(0.0, float(gap))) 

    needs_fragmentation = (num_blobs >= min_blobs_for_frag and 
                         (max(gaps_x_norm) if gaps_x_norm else 0.0) >= float(gap_threshold_norm))

    if needs_fragmentation:
        blob_metrics = {
            "needs_fragmentation": needs_fragmentation,
            "num_blobs": num_blobs,
            "blobs_norm_boxes": sorted_boxes,
            "gaps_x_norm": gaps_x_norm
            }
    # logger.info(f"Metricas: {blob_metrics}")
    return blob_metrics
        
def get_adaptive_block_size(height: float, height_thresholds: List[int], block_sizes_map: List[int]) -> int:
    for i, threshold in enumerate(height_thresholds):
        if height <= threshold:
            block_size = block_sizes_map[min(i, len(block_sizes_map) - 1)]
            return max(3, block_size if block_size % 2 != 0 else block_size + 1)
    final_block_size = block_sizes_map[-1]
    return max(3, final_block_size if final_block_size % 2 != 0 else final_block_size + 1)

def measure_polygon_quality(cropped_img: np.ndarray[Any, np.dtype[np.uint8]]) -> str:
    std = np.std(cropped_img)
    if std == 0: return "adaptive_mean"        
    hist = cv2.calcHist([cropped_img], [0], None, [256], [0, 256]).flatten()
    peaks = np.sum((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:]))
    prob = hist / np.sum(hist)
    entropy = -np.sum(prob * np.log2(prob + 1e-8))

    if peaks >= 2 and std > 30: 
        return "otsu"
        
    elif std > 20 and entropy > 5.0:
        return "adaptive_gaussian"

    elif std > 10: 
        return "sauvola"
        
    else: 
        return "adaptive_mean"
    
def otsu_binarize(cropped_img: np.ndarray[Any, np.dtype[np.uint8]]) -> np.ndarray[np.int8, Any]:
    _, bin_img = cv2.threshold(cropped_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bin_img

def adaptive_binarize(cropped_img: np.ndarray[Any, np.dtype[np.uint8]], block_size: int, c_value: int) -> np.ndarray[np.int8, Any]:
    return cv2.adaptiveThreshold(
        cropped_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, block_size, c_value
    )

def sauvola_binarize(cropped_img: np.ndarray[Any, np.dtype[np.uint8]], adaptive_block_size: int) -> np.ndarray[np.int8, Any]:
    """Sauvola thresholding producing uint8 mask with text as foreground (0) and background (255)"""
    thresh_sauvola = threshold_sauvola(cropped_img, window_size=adaptive_block_size) # type: ignore
    bin_bool: np.ndarray[np.uint8, Any] = (cropped_img > thresh_sauvola)# type: ignore
    bin_img = (bin_bool.astype(np.uint8) * 255)# type: ignore
    return bin_img# type: ignore

def adaptive_mean_fallback(cropped_img: np.ndarray[Any, np.dtype[np.uint8]], block_size: int, c_value: int) -> np.ndarray[np.int8, Any]:
    return cv2.adaptiveThreshold(
        cropped_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, block_size, max(1, c_value - 2) 
    )
