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

    def _estimate_contrast(self, gauss_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Detecta y corrige patrones de moiré en cada polígono del diccionario.
        Args:
            refined_polygons: Diccionario principal con los polígonos
        Returns:
            El mismo diccionario (moire_img), con los 'cropped_img' corregidos si aplica"""
        polygons = gauss_dict.get("polygons", {})
        for poly in polygons.values():
            cropped_img = poly.get("gauss_poly")  # Ya está correcto
            if cropped_img is not None:
                clahe_poly = self._estimate_contrast_single(cropped_img)
                poly["clahe_poly"] = clahe_poly  # Cambiar de gauss_poly a clahe_poly
                
        clahe_dict = gauss_dict
        return clahe_dict

    def _estimate_contrast_single(self, cropped_img: np.ndarray) -> np.ndarray:
        """Aplica mejora de contraste."""
        global_clahe_corrects = self.corrections.get('global', {})
        contrast_threshold = global_clahe_corrects.get('contrast_threshold', 50.0)
        clip_limit = global_clahe_corrects.get('clahe_clip_limit', 2.0)
        page_dimensions = global_clahe_corrects.get('dimension_thresholds_px', [1000, 2500])
        grid_maps = global_clahe_corrects.get('grid_sizes_map', [[6, 6], [8, 8], [10, 10]])

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
