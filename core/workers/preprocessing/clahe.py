# PerfectOCR/core/workflow/preprocessing/
import cv2
import numpy as np
import logging
import time
from typing import Dict, Any
from core.factory.abstract_worker import PreprossesingAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import CroppedImage

logger = logging.getLogger(__name__)

class ClaherEnhancer(PreprossesingAbstractWorker):

    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.corrections = config
        
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

            clahe_poly = self._estimate_contrast_single(cropped_img.cropped_img)
        else:
            return cropped_img
            
        cropped_img.cropped_img[...] = clahe_poly
        
        logger.info(f"Poligonos corregidos Clahe {clahe_poly}")
        total_time = time.time() - start_time
        logger.info(f"Clahe completado en: {total_time:.3f}s")

        return cropped_img

    def _estimate_contrast_single(self, cropped_img: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Aplica mejora de contraste."""
        global_clahe_corrects = self.corrections.get('global', {})
        contrast_threshold = global_clahe_corrects.get('contrast_threshold', 50.0)
        clip_limit = global_clahe_corrects.get('clahe_clip_limit', 2.0)
        page_dimensions = global_clahe_corrects.get('dimension_thresholds_px', [1000, 2500])
        grid_maps = global_clahe_corrects.get('grid_sizes_map', [[6, 6], [8, 8], [10, 10]])

        clahe_poly = []

        # Cálculo de estadísticos:
        contrast_std = np.std(cropped_img) 
        global_var = np.var(cropped_img)
        max_value = np.max(cropped_img)
        min_value = np.min(cropped_img)
        dynamic_range = max_value - min_value
        dynamic_interval = max(30.0, global_var * 0.6)
        adaptative_threshold = dynamic_interval if dynamic_range > contrast_threshold else 20
                
        if contrast_std < adaptative_threshold:
            img_h, img_w = cropped_img.shape
            max_dim = max(img_h, img_w)
            if max_dim < page_dimensions[0]:
                grid_size = grid_maps[0]
            elif max_dim < page_dimensions[1]:
                grid_size = grid_maps[1]
            else:
                grid_size = grid_maps[2]
            clip_limit = min(3.0, max(1.0, dynamic_range * 0.01))

            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
            clahe_poly = clahe.apply(cropped_img)
        else: 
            clahe_poly = cropped_img

        return clahe_poly
