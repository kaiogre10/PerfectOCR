# PerfectOCR/core/workflow/preprocessing/
import cv2
import numpy as np
import logging
from typing import Dict, Any
from core.factory.abstract_worker import AbstractWorker
from core.domain.data_formatter import DataFormatter

logger = logging.getLogger(__name__)

class GaussianDenoiser(AbstractWorker):

    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.config = config
        
    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """Detecta y corrige patrones de moiré en cada polígono del diccionario.
        Args:
            refined_polygons: Diccionario principal con los polígonos
        Returns:
            El mismo diccionario (moire_img), con los 'cropped_img' corregidos si aplica"""
        polygons = context.get("polygons", {})
        for poly_data in polygons.values():
            current_image = poly_data.get("cropped_img")
            if current_image is not None:
                # Procesa la imagen y la sobrescribe en el mismo lugar
                poly_data["cropped_img"] = self._estimate_gaussian_noise_single(current_image)
        


    def _estimate_gaussian_noise_single(self,  cropped_img: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Estima ruido general con varianza del Laplaciano."""
        bilateral_corrections = self.config.get('bilateral_params', {})
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