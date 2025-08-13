# PerfectOCR/core/workflow/preprocessing/sharp.py
import cv2
import time
import numpy as np
import logging
from typing import Dict, Any, Optional
from skimage import filters 
from core.factory.abstract_worker import PreprossesingAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import CroppedImage

logger = logging.getLogger(__name__)

class SharpeningEnhancer(PreprossesingAbstractWorker):

    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.corrections = config

    def preprocess(self, cropped_img: CroppedImage, manager: DataFormatter) -> CroppedImage:
        """
        Detecta y corrige patrones de moiré en cada polígono del diccionario,
        modificando 'cropped_img' in-situ.
        """
        start_time = time.time()
        
        img = cropped_img.cropped_img
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        sharp_poly = self._estimate_sharpness_single(img)
            
        cropped_img.cropped_img = sharp_poly
        
        total_time = time.time() - start_time
        logger.debug(f"Sharp completado en: {total_time:.3f}s")

        return cropped_img

    def _estimate_sharpness_single(self, current_image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Estima nitidez con Sobel."""
        sharpen_corrections = self.corrections.get('sharpening', {})
        radius = sharpen_corrections.get('radius', 1.0)
        amount = sharpen_corrections.get('amount', 1.5)

        # Calcular Sobel y otros estadísticos
        sobel = cv2.Sobel(current_image, cv2.CV_64F, 1, 1, ksize=3)
        sharpness = np.mean(np.abs(sobel))

        global_sharp_var = np.var(current_image)
        adaptative_sharp_threshold = max(30, global_sharp_var * 0.5)
        
        if sharpness < adaptative_sharp_threshold:
            radius = min(2.0, max(0.5, global_sharp_var - 0.02))
            amount = min(2.0, max(1.0, global_sharp_var - 0.03))
            

            sharpened: Optional[np.ndarray[Any, Any]] = filters.unsharp_mask(current_image, radius=float(radius), amount=float(amount))

            if sharpened is not None:
                sharp_poly = (sharpened * 255).astype(np.uint8)
            else:
                sharp_poly = current_image
        
        else:
            sharp_poly = current_image

        return sharp_poly