# PerfectOCR/core/polygonal/binarization.py
from sklearnex import patch_sklearn
patch_sklearn()
import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from skimage.filters import threshold_sauvola
from skimage.util import img_as_ubyte

logger = logging.getLogger(__name__)

class Binarizator:
    
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.geometric = config
        self.geometric_bin = config.get('polygon_config', {})
        self.binarize_config = config.get('binarize', {})
        self.c_value = self.binarize_config.get('c_value', 7)
        self.height_thresholds = self.binarize_config.get('height_thresholds_px', [100, 800, 1500, 2500])
        self.block_sizes_map = self.binarize_config.get('block_sizes_map', [15, 21, 25, 35, 41])
        
    def _measure_polygon_quality(self, gray_img: np.ndarray) -> str:
        """
        Evalúa la calidad local del polígono y decide el método de binarización.
        Retorna el nombre del método a usar.
        """
        std = np.std(gray_img)
        hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256]).flatten()
        peaks = np.sum((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:]))
        prob = hist / np.sum(hist)
        entropy = -np.sum(prob * np.log2(prob + 1e-8))

        if peaks >= 2 and std > 30:
            return "otsu"
        elif std > 20 and entropy > 4.5:
            return "adaptive_gaussian"
        elif std > 10:
            return "sauvola"
        else:
            return "adaptive_mean"

    def _get_adaptive_block_size(self, height: float) -> int:
        """Determina el block_size adaptativo basado en la altura del polígono."""
        for i, threshold in enumerate(self.height_thresholds):
            if height <= threshold:
                
                block_size = self.block_sizes_map[min(i, len(self.block_sizes_map) - 1)]
                return max(3, block_size if block_size % 2 != 0 else block_size + 1)
            
        final_block_size = self.block_sizes_map[-1]
        adaptive_block_size = max(3, final_block_size if final_block_size % 2 != 0 else final_block_size + 1)
    
        return adaptive_block_size
        
    def _otsu_binarize(self, gray_img: np.ndarray) -> np.ndarray:
        """Binarización Otsu."""
        _, otsu_result = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return otsu_result
       
    def _adaptive_binarize(self, gray_img: np.ndarray, block_size: int) -> np.ndarray:
        """Binarización adaptativa Gaussiana."""
        return cv2.adaptiveThreshold(
            gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, block_size, self.c_value
        )
    
    def _sauvola_binarize(self, gray_img: np.ndarray, adaptive_block_size: int) -> np.ndarray:
        """Binarización Sauvola de scikit-image."""
        thresh_sauvola = threshold_sauvola(gray_img, adaptive_block_size)
        return img_as_ubyte(gray_img > thresh_sauvola)

    def _adaptive_mean_fallback(self, gray_img: np.ndarray, block_size: int) -> np.ndarray:
        """Fallback con adaptive mean de OpenCV."""
        return cv2.adaptiveThreshold(
            gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, block_size, max(1, self.c_value - 2)
        )
   
    def _process_individual_polygons(self, gray_img: np.ndarray, height: float) -> np.ndarray:
        """Binarización basada en una sola medición de calidad local."""
        adaptive_block_size = self._get_adaptive_block_size(height)
        metodo = self._measure_polygon_quality(gray_img)
        logger.debug(f"Selección de método '{metodo}' para polígono de altura {height}")

        if metodo == "otsu":
            binarized_polygons = self._otsu_binarize(gray_img)
        elif metodo == "adaptive_gaussian":
            binarized_polygons = self._adaptive_binarize(gray_img, adaptive_block_size)
        elif metodo == "sauvola":
            binarized_polygons = self._sauvola_binarize(gray_img, adaptive_block_size)
        else:
            binarized_polygons = self._adaptive_mean_fallback(gray_img, adaptive_block_size)

        return binarized_polygons
    
    def _binarize_polygons(self, polygons_to_bin: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Procesa un diccionario de polígonos individuales aplicando binarización adaptativa a cada uno. Devuelve un diccionario con polygon_id -> binarized_img"""
        if not polygons_to_bin:
            logger.warning("Diccionario de polígonos vacío recibido")
            return {}

        binarized_polygons = {}
        binarized_count = 0
        failed_count = 0

        for polygon_id, polygon_data in polygons_to_bin.items():
            try:
                cropped_img = polygon_data.get("cropped_img")
                if cropped_img is None:
                    logger.warning(f"Polígono {polygon_id} sin imagen, omitiendo")
                    failed_count += 1
                    continue

                height = cropped_img.shape[0]
                gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY) if len(cropped_img.shape) == 3 else cropped_img
                bin_img = self._process_individual_polygons(gray_img, height)

                # Guardar solo el polygon_id y la imagen binarizada
                binarized_polygons[polygon_id] = bin_img
                binarized_count += 1

            except Exception as e:
                logger.error(f"Error procesando polígono {polygon_id}: {e}")
                failed_count += 1
        
        total_processed = len(polygons_to_bin)
        logger.info(f"Binarización completada: {binarized_count}/{total_processed} polígonos procesados.")

        return binarized_polygons