# PerfectOCR/core/workflow/preprocessing/sp.py
import cv2
import numpy as np
import logging
import time
from typing import Dict, Any 
from core.factory.abstract_worker import PreprossesingAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import CroppedImage
logger = logging.getLogger(__name__)
    
class DoctorSaltPepper(PreprossesingAbstractWorker):
    total_polygons_corrected = 0
    
    def __init__(self, config: Dict[str, Any], project_root: str):    
        self.project_root = project_root
        self.config = config
    
    def preprocess(self, cropped_img: CroppedImage, manager: DataFormatter) -> CroppedImage:
        """
        Detecta y corrige patrones de sal y pimienta en cada polígono del diccionario,
        modificando 'cropped_img' in-situ.
        """
        start_time = time.time()
        if not isinstance(cropped_img.cropped_img, np.ndarray): # type: ignore
            cropped_img.cropped_img = np.array(cropped_img.cropped_img)
        if cropped_img.cropped_img.dtype != np.uint8:
            cropped_img.cropped_img = cropped_img.cropped_img.astype(np.uint8)
        self._detect_sp_single(cropped_img.cropped_img)
        
        DoctorSaltPepper.total_polygons_corrected += 1
        total_time = time.time() - start_time
        logger.debug(f"S&P completado en: {total_time:.4}s")
        logger.debug(f"Total de polígonos corregidos S&P: {DoctorSaltPepper.total_polygons_corrected}")
        
        return cropped_img

    
    def _detect_sp_single(self, cropped_img: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Estima ruido sal y pimienta con histograma, operando siempre in-place sobre el array original."""
        sp_corrections = self.config.get('median_filter', {})
        low_thresh = int(sp_corrections.get('salt_pepper_low', 10))
        high_thresh = int(sp_corrections.get('salt_pepper_high', 245))
        kernel_size = int(sp_corrections.get('kernel_size', 3))
        sp_threshold = float(sp_corrections.get('salt_pepper_threshold', 0.001))

        total_pixels = cropped_img.size
        if total_pixels == 0:
            return cropped_img

        # Calcular histograma
        hist, _ = np.histogram(cropped_img, bins=256, range=(0, 255))

        # Sumar píxeles en los extremos del histograma
        sp_pixels = int(np.sum(hist[:low_thresh]) + np.sum(hist[high_thresh:]))
        sp_ratio = sp_pixels / total_pixels

        if sp_ratio > sp_threshold:
            # Ajustar kernel_size a impar si es necesario
            if kernel_size % 2 == 0:
                kernel_size += 1
            # Aplicar filtro de mediana y copiar el resultado sobre el array original (in-place)
            filtered = cv2.medianBlur(cropped_img, kernel_size)
            if filtered.shape == cropped_img.shape and filtered.dtype == cropped_img.dtype:
                cropped_img[...] = filtered
            else:
                np.copyto(cropped_img, filtered, casting='unsafe')
        # Si no hay suficiente ruido, no se modifica la imagen

        return cropped_img