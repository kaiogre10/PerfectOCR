# PerfectOCR/core/workflow/preprocessing/
import cv2
import numpy as np
import logging
from typing import Dict, Any
from core.factory.abstract_worker import PreprossesingAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import CroppedImage
import time

logger = logging.getLogger(__name__)

class GaussianDenoiser(PreprossesingAbstractWorker):

    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.config = config
        
    def preprocess(self, cropped_img: CroppedImage, manager: DataFormatter) -> CroppedImage:
        """
        Detecta y corrige patrones de moiré en cada polígono del diccionario,
        modificando 'cropped_img' in-situ.
        """
        start_time = time.time()

        if not isinstance(cropped_img.cropped_img, np.ndarray): # type: ignore
            cropped_img.cropped_img = np.array(cropped_img.cropped_img)
        if cropped_img.cropped_img.dtype != np.uint8:
            cropped_img.cropped_img = cropped_img.cropped_img.astype(np.uint8)
            gauss_poly = self._estimate_gaussian_noise_single(cropped_img.cropped_img)
        else:
            return cropped_img
            
        cropped_img.cropped_img[...] = gauss_poly
        
        logger.debug(f"Poligonos corregidos Gauss {gauss_poly}")
        total_time = time.time() - start_time
        logger.debug(f"Gauss completado en: {total_time:.3f}s")

        return cropped_img

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