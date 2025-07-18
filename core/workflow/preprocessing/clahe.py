# PerfectOCR/core/workflow/preprocessing/
import cv2
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ClaherEnhancer:

    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.corrections = config

    def _estimate_contrast(self, gauss_img: np.ndarray) -> np.ndarray:
        """Aplica mejora de contraste."""
        global_clahe_corrects = self.corrections.get('global', {})
        contrast_threshold = global_clahe_corrects.get('contrast_threshold', 50.0)
        clip_limit = global_clahe_corrects.get('clahe_clip_limit', 2.0)
        page_dimensions = global_clahe_corrects.get('dimension_thresholds_px', [1000, 2500])
        grid_maps = global_clahe_corrects.get('grid_sizes_map', [[6, 6], [8, 8], [10, 10]])

        # Cálculo de estadísticos:
        contrast_std = np.std(gauss_img) 
        global_var = np.var(gauss_img)
        max_value = np.max(gauss_img)
        min_value = np.min(gauss_img)
        dynamic_range = max_value - min_value
        dynamic_interval = max(30.0, global_var * 0.6)
        adaptative_threshold = dynamic_interval if dynamic_range > contrast_threshold else 20
                
        if contrast_std < adaptative_threshold:
            img_h, img_w = gauss_img.shape
            max_dim = max(img_h, img_w)
            if max_dim < page_dimensions[0]:
                grid_size = grid_maps[0]
            elif max_dim < page_dimensions[1]:
                grid_size = grid_maps[1]
            else:
                grid_size = grid_maps[2]
            clip_limit = min(3.0, max(1.0, dynamic_range * 0.01))

            logger.info(f"-> Aplicando mejora de contraste: CLAHE (clip_limit={clip_limit}, grid_size={grid_size}).")
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
            clahed_img = clahe.apply(gauss_img)
        else: 
            clahed_img = gauss_img

        return clahed_img
