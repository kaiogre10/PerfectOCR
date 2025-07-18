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

    def _estimate_salt_pepper_noise(self, moire_img: np.ndarray) -> np.ndarray:
        """Estima ruido sal y pimienta con histograma."""
        sp_corrections = self.denoise_corrections.get('median_filter', {})
        low_thresh = sp_corrections.get('salt_pepper_low', 10)
        high_thresh = sp_corrections.get('salt_pepper_high', 245)
        kernel_size = sp_corrections.get('kernel_size', 3)
        sp_threshold = sp_corrections.get('salt_pepper_threshold', 0.001)

        # Medición del ruido
        total_pixels = moire_img.size
        if total_pixels == 0:
            return moire_img
        
        hist, _ = histogram(moire_img, nbins=256, source_range='image')
        
        # Sumar píxeles en extremos
        sp_pixels = np.sum(hist[:low_thresh]) + np.sum(hist[high_thresh:])
        sp_ratio = sp_pixels / total_pixels
        logger.info(f"Ruido S&P medido: {sp_ratio:.2%}")

        if sp_ratio > sp_threshold:
            logger.info(f"APLICANDO FILTRO MEDIANA: {sp_ratio:.4f}")
            # Si el kernel_size es par, lo ajustamos a impar
            if kernel_size % 2 == 0:
                kernel_size += 1
            sp_img = cv2.medianBlur(moire_img, kernel_size)
        else:
            sp_img = moire_img
        
        return sp_img