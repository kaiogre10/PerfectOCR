# PerfectOCR/core/workflow/preprocessing/sp.py
import cv2
import numpy as np
import logging
from typing import Dict, Any, Optional
from skimage.exposure import histogram
from core.factory.abstract_worker import AbstractWorker
from core.domain.data_formatter import DataFormatter


logger = logging.getLogger(__name__)
    
class DoctorSaltPepper(AbstractWorker):

    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.config = config
    
    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:

        polygons = context.get("polygons", {})
        for poly_data in polygons.values():
            current_image = poly_data.get("cropped_img")
            if current_image is not None:
                # Procesa la imagen y la sobrescribe en el mismo lugar
                poly_data["cropped_img"] = self._detect_sp_single(current_image)
                
    
    def _detect_sp_single(self, cropped_img: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Estima ruido sal y pimienta con histograma."""
        sp_corrections = self.config.get('median_filter', {})
        low_thresh = sp_corrections.get('salt_pepper_low', 10)
        high_thresh = sp_corrections.get('salt_pepper_high', 245)
        kernel_size = sp_corrections.get('kernel_size', 3)
        sp_threshold = sp_corrections.get('salt_pepper_threshold', 0.001)
        
        sp_poly = []

        # Medición del ruido
        total_pixels = cropped_img.size
        if total_pixels == 0:
            sp_poly =  cropped_img
            return sp_poly

        hist, _ = histogram(cropped_img, nbins=256, source_range='cropped_img')
        
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