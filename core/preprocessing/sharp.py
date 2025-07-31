# PerfectOCR/core/workflow/preprocessing/sharp.py
from sklearnex import patch_sklearn
patch_sklearn()
import cv2
import numpy as np
import logging
from typing import Dict, Any
from skimage.filters import threshold_sauvola, unsharp_mask


logger = logging.getLogger(__name__)

class SharpeningEnhancer:

    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.corrections = config

    def _estimate_sharpness(self, clahed_img: np.ndarray) -> np.ndarray:
        """Estima nitidez con Sobel."""
        sharpen_corrections = self.corrections.get('sharpening', {})
        radius = sharpen_corrections.get('radius', 1.0)
        amount = sharpen_corrections.get('amount', 1.5)

        # Calcular Sobel y otros estadísticos
        sobel = cv2.Sobel(clahed_img, cv2.CV_64F, 1, 1, ksize=3)
        sharpness = np.mean(np.abs(sobel))

        global_sharp_var = np.var(clahed_img)
        adaptative_sharp_threshold = max(30, global_sharp_var * 0.5)
        
        if sharpness < adaptative_sharp_threshold:
            radius = min(2.0, max(0.5, global_sharp_var - 0.02))
            amount = min(2.0, max(1.0, global_sharp_var - 0.03))

            sharpened = unsharp_mask(clahed_img, radius=radius, amount=amount)
            corrected_imag = (sharpened * 255).astype(np.uint8)
        else:
            corrected_imag = clahed_img

        return corrected_imag