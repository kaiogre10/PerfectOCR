# PerfectOCR/core/workflow/preprocessing/cleanner.py
import cv2
import logging
from typing import Any, Dict
import numpy as np

logger = logging.getLogger(__name__)

class ImageCleaner:

    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.corrections = config

    def _quick_enhance(self, gray_image: np.ndarray) -> np.ndarray:
        """Aplica una secuencia de mejoras rápidas y adaptativas a una imagen en escala de grises."""
        quick_corrections = self.corrections
        
        # --- 1. Denoising Adaptativo (Bilateral Filter) ---
        img_var = np.var(gray_image)
        if img_var < 100:
            d, sigma_color, sigma_space = 5, 30, 30
        else:
            d, sigma_color, sigma_space = 9, 60, 60
        denoised = cv2.bilateralFilter(gray_image, d, sigma_color, sigma_space)
        
        # --- 2. Mejora de Contraste Adaptativo (CLAHE) ---
        img_std = np.std(denoised)
        if img_std < 25:
            clip_limit = 3.0
            grid_size = (6, 6)
        elif img_std < 50:
            clip_limit = 2.0
            grid_size = (8, 8)
        else:
            clip_limit = 1.0
            grid_size = (10, 10)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        enhanced = clahe.apply(denoised)
        
        # --- 3. Aumento de Nitidez (Sharpening) Adaptativo ---
        mean_intensity = np.mean(enhanced)
        if mean_intensity < 128:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        else:
            kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
        
        clean_img = cv2.filter2D(enhanced, -1, kernel)
        
        logger.info(f"Quick enhance aplicado: Varianza={img_var:.1f}, Desv. Est.={img_std:.1f}, Intensidad Media={mean_intensity:.1f}")
        
        return clean_img
