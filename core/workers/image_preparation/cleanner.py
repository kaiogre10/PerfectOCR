# PerfectOCR/core/workers/image_preparation/cleanner.py
import cv2
import logging
from typing import Dict, Any, Optional
import numpy as np
from core.factory.abstract_worker import ImagePrepAbstractWorker
from core.domain.data_formatter import DataFormatter

logger = logging.getLogger(__name__)

class ImageCleaner(ImagePrepAbstractWorker):

    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.corrections = config
        
    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        try:
            full_img: Optional[np.ndarray[Any, Any]] = context.get("full_img")
            if full_img is None:
                logger.error("Cleaner: full_img no encontrado en contexto")
                return False

            # Convertir a escala de grises si es color
            if len(full_img.shape) == 3 and full_img.shape[2] == 3:
                img_gray = cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = full_img.copy()

            # 1) bilateral adaptativo
            img_var = float(np.var(img_gray))
            d, sC, sS = (5, 30, 30) if img_var < 100 else (9, 60, 60)
            denoised = cv2.bilateralFilter(img_gray, d, sC, sS)

            # 2) CLAHE adaptativo
            img_std = float(np.std(denoised))
            if img_std < 25:
                clip, grid = 3.0, (6, 6)
            elif img_std < 50:
                clip, grid = 2.0, (8, 8)
            else:
                clip, grid = 1.0, (10, 10)
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
            enhanced = clahe.apply(denoised)

            # 3) sharpen adaptativo
            mean_intensity = float(np.mean(enhanced))
            if mean_intensity < 128:
                kernel = np.array([[0, -1, 0], [-1, 3, -1], [0, -1, 0]])
            else:
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            clean_img = cv2.filter2D(enhanced, -1, kernel)

            # Normalización final
            clean_img = np.clip(clean_img, 0, 255).astype(np.uint8)

            full_img[...] = clean_img
            logger.info("Cleaner: limpieza automatizada aplicada correctamente.")
            return True
        except Exception as e:
            logger.error(f"Cleaner: {e}", exc_info=True)
            return False