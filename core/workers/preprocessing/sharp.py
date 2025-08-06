# PerfectOCR/core/workflow/preprocessing/sharp.py
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

    def _estimate_sharpness(self, processing_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Detecta y corrige patrones de moiré en cada polígono del diccionario.
        Args:
            refined_polygons: Diccionario principal con los polígonos
        Returns:
            El mismo diccionario (moire_img), con los 'cropped_img' corregidos si aplica"""
        polygons = processing_dict.get("polygons", {})
        for poly_data in polygons.values():
            current_image = poly_data.get("cropped_img")
            if current_image is not None:
                # Procesa la imagen y la sobrescribe en el mismo lugar
                poly_data["cropped_img"] = self._estimate_sharpness_single(current_image)
        
        return processing_dict


    def _estimate_sharpness_single(self, clahed_img: np.ndarray) -> np.ndarray:
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

            sharpened = unsharp_mask(clahed_img, radius=float(radius), amount=float(amount))
            sharp_poly = (sharpened * 255).astype(np.uint8)
        else:
            sharp_poly = clahed_img

        return sharp_poly