# PerfectOCR/core/workflow/preprocessing/sp.py
import cv2
import numpy as np
import logging
from typing import Dict, Any
from skimage.exposure import histogram

logger = logging.getLogger(__name__)
    
class DoctorSaltPepper:

    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.corrections = config
        self.denoise_corrections = config.get('denoise', {})
    
    def _estimate_salt_pepper_noise(self, moire_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Detecta y corrige patrones de moiré en cada polígono del diccionario.
        Args:
            refined_polygons: Diccionario principal con los polígonos
        Returns:
            El mismo diccionario (moire_img), con los 'cropped_img' corregidos si aplica"""
        polygons = moire_dict.get("polygons", {})
        for poly in polygons.values():
            cropped_img = poly.get("moire_poly")
            if cropped_img is not None:
                sp_poly = self._detect_sp_single(cropped_img)
                poly["sp_poly"] = sp_poly
        sp_dict = moire_dict
        return sp_dict
                
    
    def _detect_sp_single(self, cropped_img: np.ndarray) -> np.ndarray:
        """Estima ruido sal y pimienta con histograma."""
        sp_corrections = self.denoise_corrections.get('median_filter', {})
        low_thresh = sp_corrections.get('salt_pepper_low', 10)
        high_thresh = sp_corrections.get('salt_pepper_high', 245)
        kernel_size = sp_corrections.get('kernel_size', 3)
        sp_threshold = sp_corrections.get('salt_pepper_threshold', 0.001)

        # Medición del ruido
        total_pixels = cropped_img.size
        if total_pixels == 0:
            sp_poly =  cropped_img
            return sp_poly
        
        hist, _ = histogram(cropped_img, nbins=256, source_range='image')
        
        # Sumar píxeles en extremos
        sp_pixels = np.sum(hist[:low_thresh]) + np.sum(hist[high_thresh:])
        sp_ratio = sp_pixels / total_pixels

        if sp_ratio > sp_threshold:
            # Si el kernel_size es par, lo ajustamos a impar
            if kernel_size % 2 == 0:
                kernel_size += 1
            sp_poly = cv2.medianBlur(cropped_img, kernel_size)
        else:
            sp_poly = cropped_img
        
        return sp_poly