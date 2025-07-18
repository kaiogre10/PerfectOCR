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

    def _estimate_gaussian_noise(self, sp_img: np.ndarray) -> np.ndarray:
        """Estima ruido general con varianza del Laplaciano."""
        bilateral_corrections = self.corrections.get('bilateral_params', {})
        d = bilateral_corrections.get('d', 9)
        sigma_color = bilateral_corrections.get('sigma_color', 75)
        sigma_space = bilateral_corrections.get('sigma_space', 75)
        gauss_threshold = bilateral_corrections.get('laplacian_variance_threshold', 100)

        # Medición 
        laplacian_var = cv2.Laplacian(sp_img, cv2.CV_64F).var()
        logger.info(f"Ruido general (varianza laplaciana): {laplacian_var:.3f}")
        if laplacian_var > gauss_threshold:
            return sp_img
        
        if laplacian_var < gauss_threshold:
            logger.info(f"-> Aplicando filtro de ruido general: Bilateral (d={d}, sigmaColor={sigma_color}).")
            gauss_img = cv2.bilateralFilter(sp_img, d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
        else:
            gauss_img = sp_img

        return gauss_img