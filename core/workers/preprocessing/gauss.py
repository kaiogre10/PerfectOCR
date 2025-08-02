# PerfectOCR/core/workflow/preprocessing/
import cv2
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class GaussianDenoiser:

    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.corrections = config
        self.denoise_corrections = config.get('denoise', {})

    def _estimate_gaussian_noise(self, sp_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Detecta y corrige patrones de moiré en cada polígono del diccionario.
        Args:
            refined_polygons: Diccionario principal con los polígonos
        Returns:
            El mismo diccionario (moire_img), con los 'cropped_img' corregidos si aplica"""
        polygons = sp_dict.get("polygons", {})
        for poly in polygons.values():
            cropped_img = poly.get("sp_poly")  # Ya está correcto
            if cropped_img is not None:
                gauss_poly = self._estimate_gaussian_noise_single(cropped_img)
                poly["gauss_poly"] = gauss_poly  # Cambiar de sp_poly a gauss_poly
                
        gauss_dict = sp_dict
        return gauss_dict


    def _estimate_gaussian_noise_single(self, cropped_img: np.ndarray) -> np.ndarray:
        """Estima ruido general con varianza del Laplaciano."""
        bilateral_corrections = self.corrections.get('bilateral_params', {})
        d = bilateral_corrections.get('d', 9)
        sigma_color = bilateral_corrections.get('sigma_color', 75)
        sigma_space = bilateral_corrections.get('sigma_space', 75)
        gauss_threshold = bilateral_corrections.get('laplacian_variance_threshold', 100)

        laplacian_var = cv2.Laplacian(cropped_img, cv2.CV_64F).var()
        if laplacian_var > gauss_threshold:
            gauss_poly = cropped_img
            return gauss_poly
        
        if laplacian_var < gauss_threshold:
            gauss_poly = cv2.bilateralFilter(cropped_img, d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
        else:
            gauss_poly = cropped_img

        return gauss_poly