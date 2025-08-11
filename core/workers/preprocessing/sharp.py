# PerfectOCR/core/workflow/preprocessing/sharp.py
import cv2
import numpy as np
import logging
from typing import Dict, Any, Optional
from skimage.filters import unsharp_mask  # type: ignore
from core.factory.abstract_worker import AbstractWorker
from core.domain.data_formatter import DataFormatter

logger = logging.getLogger(__name__)

class SharpeningEnhancer(AbstractWorker):

    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.corrections = config

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
                poly_data["cropped_img"] = self._estimate_sharpness_single(current_image)


    def _estimate_sharpness_single(self, cropped_img: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Estima nitidez con Sobel."""
        sharpen_corrections = self.corrections.get('sharpening', {})
        radius = sharpen_corrections.get('radius', 1.0)
        amount = sharpen_corrections.get('amount', 1.5)

        sharp_poly = []

        # Calcular Sobel y otros estadísticos
        sobel = cv2.Sobel(cropped_img, cv2.CV_64F, 1, 1, ksize=3)
        sharpness = np.mean(np.abs(sobel))

        global_sharp_var = np.var(cropped_img)
        adaptative_sharp_threshold = max(30, global_sharp_var * 0.5)
        
        if sharpness < adaptative_sharp_threshold:
            radius = min(2.0, max(0.5, global_sharp_var - 0.02))
            amount = min(2.0, max(1.0, global_sharp_var - 0.03))

            sharpened: Optional[np.ndarray[Any, Any]] = unsharp_mask(cropped_img, radius=float(radius), amount=float(amount))

            sharpened = np.asarray(sharpened, None)

            sharp_poly = (sharpened * 255).astype(np.uint8)
        
        else:
            sharp_poly = cropped_img

        return sharp_poly