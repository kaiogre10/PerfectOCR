# PerfectOCR/core/polygonal/binarization.py
from sklearnex import patch_sklearn
patch_sklearn()
import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from skimage.measure import regionprops, label
from skimage.filters import threshold_sauvola
from skimage.util import img_as_ubyte
from skimage import morphology

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

    def _get_adaptive_block_size(self, height: float) -> int:
        """Determina el block_size adaptativo basado en la altura del polígono."""
        for i, threshold in enumerate(self.height_thresholds):
            if height <= threshold:
                
                block_size = self.block_sizes_map[min(i, len(self.block_sizes_map) - 1)]
                return max(3, block_size if block_size % 2 != 0 else block_size + 1)
            
        final_block_size = self.block_sizes_map[-1]
        adaptive_block_size = max(3, final_block_size if final_block_size % 2 != 0 else final_block_size + 1)
    
        return adaptive_block_size
    
    def _is_histogram_bimodal(self, gray_img: np.ndarray) -> bool:
        """Estima si el histograma es bimodal, bueno para Otsu."""
        hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
        hist = hist.flatten()
        
        peaks = []
        for i in range(1, 255):
            if hist[i-1] < hist[i] and hist[i+1] < hist[i]:
                peaks.append(hist[i])
        
        return len(peaks) >= 2
    
    def _otsu_binarize(self, gray_img: np.ndarray) -> np.ndarray:
        """Binarización Otsu."""
        _, otsu_result = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return otsu_result

    def _check_quality(self, otsu_result: np.ndarray) -> bool:
        """Verifica calidad basada en ratio de píxeles blancos."""
        qa_params = self.binarize_config.get('quality', {})
        quality_min = qa_params.get('quality_min', 0.005)  
        quality_max = qa_params.get('quality_max', 0.95)   
    
        if otsu_result is None or otsu_result.size == 0:
            return False
        
        white_ratio = np.sum(otsu_result == 255) / otsu_result.size
        return quality_min < white_ratio < quality_max
       
    def _adaptive_binarize(self, gray_img: np.ndarray, block_size: int) -> np.ndarray:
        """Binarización adaptativa Gaussiana."""
        return cv2.adaptiveThreshold(
            gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, block_size, self.c_value
        )
    
    def _sauvola_binarize(self, gray_img: np.ndarray, window_size: int) -> np.ndarray:
        """Binarización Sauvola de scikit-image."""
        thresh_sauvola = threshold_sauvola(gray_img, window_size=window_size)
        return img_as_ubyte(gray_img > thresh_sauvola)

    def _adaptive_mean_fallback(self, gray_img: np.ndarray, block_size: int) -> np.ndarray:
        """Fallback con adaptive mean de OpenCV."""
        return cv2.adaptiveThreshold(
            gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, block_size, max(1, self.c_value - 2)
        )
   
    def _process_individual_polygons(self, gray_img: np.ndarray, height: float) -> np.ndarray:
        """Binarización robusta y adaptativa para un polígono individual."""
            
        adaptive_block_size = self._get_adaptive_block_size(height)
        logger.debug(f"Usando block_size adaptativo {adaptive_block_size} para polígono de altura {height}")
        
        if self._is_histogram_bimodal(gray_img):
            try:
                otsu_result = self._otsu_binarize(gray_img)
                if self._check_quality(otsu_result):
                    logger.debug("Binarización exitosa con método Otsu")
                    return otsu_result
            except Exception as e:
                logger.warning(f"Error en binarización Otsu: {e}")
       
        try:
            adaptive_result = self._adaptive_binarize(gray_img, adaptive_block_size)
            if self._check_quality(adaptive_result):
                logger.debug("Binarización exitosa con método Adaptive Gaussian")
                return adaptive_result
        except Exception as e:
            logger.warning(f"Error en binarización Adaptive Gaussian: {e}")

        try:
            sauvola_result = self._sauvola_binarize(gray_img, adaptive_block_size)
            if self._check_quality(sauvola_result):
                logger.debug("Binarización exitosa con método Sauvola (skimage)")
                return sauvola_result
        except Exception as e:
            logger.warning(f"Error en binarización Sauvola: {e}")
       
        try:
            fallback_result = self._adaptive_mean_fallback(gray_img, adaptive_block_size)
            logger.debug("Usando método fallback Adaptive Mean")
            return fallback_result
        except Exception as e:
            logger.error(f"Error en método fallback final: {e}")
            return gray_img
    
    def _clean_binarizated_polys(self, binarized_polygons: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Limpia cada imagen binarizada de los polígonos individuales.
        Versión adaptada para diccionario.
        """
        cleaned_polygons = {}
        
        for polygon_id, binary_img in binarized_polygons.items():
            if binary_img is None or not isinstance(binary_img, np.ndarray) or binary_img.size == 0:
                continue

            try:
                labeled_img = label(binary_img.astype(bool))
                if isinstance(labeled_img, tuple):
                    labeled_img = labeled_img[0]
                elif isinstance(labeled_img, list):
                    labeled_img = np.array(labeled_img)
                elif not isinstance(labeled_img, np.ndarray):
                    labeled_img = np.array(labeled_img)
                num_regions = int(labeled_img.max()) if labeled_img.size > 0 else 0
                
                if num_regions == 0:
                    continue
            except Exception as e:
                logger.error(f"Ocurrió una excepción en label para polígono {polygon_id}: {e}")
                continue
        
            props = regionprops(labeled_img)
            if not props:
                continue

            areas = [prop.area for prop in props]
            median_area = np.median(areas)
            adaptive_min_area = median_area * 0.1
            umbral_excentricidad = 0.99
            umbral_solidez = 0.5

            img_limpia = np.zeros_like(binary_img, dtype=np.uint8)
            for prop in props:
                if (prop.area >= adaptive_min_area and 
                    prop.eccentricity < umbral_excentricidad and 
                    prop.solidity > umbral_solidez):
                    for y, x in prop.coords:
                        img_limpia[y, x] = 255
    
            cleaned_polygons[polygon_id] = img_limpia

        return cleaned_polygons
    
    def _binarize_polygons(self, polygons_to_bin: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Procesa un diccionario de polígonos individuales aplicando binarización adaptativa a cada uno.
        Devuelve un diccionario con polygon_id -> binarized_img
        """
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

                padding_coords = polygon_data.get("padding_coords")
                if padding_coords is not None and len(padding_coords) == 4:
                    height = padding_coords[3] - padding_coords[1]
                else:
                    logger.info("No padding_coords válidas")
                    height = cropped_img.shape[0]
                    
                gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY) if len(cropped_img.shape) == 3 else cropped_img
                bin_img = self._process_individual_polygons(gray_img, height)
                
                # Guardar solo el polygon_id y la imagen binarizada
                binarized_polygons[polygon_id] = bin_img
                binarized_count += 0
                
            except Exception as e:
                logger.error(f"Error procesando polígono {polygon_id}: {e}")
                failed_count += 1

        cleaned_binarized_polygons = self._clean_binarizated_polys(binarized_polygons)
        cleaned_count = len(cleaned_binarized_polygons)

        logger.info(f"Binarización completada: {binarized_count}/{len(polygons_to_bin)} polígonos binarizados exitosamente")
        logger.info(f"Polígonos limpios tras post-procesamiento: {cleaned_count}/{binarized_count}")

        return cleaned_binarized_polygons
