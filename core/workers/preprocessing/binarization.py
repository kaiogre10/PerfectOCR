# PerfectOCR/core/polygonal/binarization.py
import cv2
import numpy as np
import logging
from typing import Dict, Any, Optional
from skimage.filters import threshold_sauvola # type: ignore
from skimage.util import img_as_ubyte # type: ignore
from core.factory.abstract_worker import AbstractWorker
from core.domain.data_formatter import DataFormatter

logger = logging.getLogger(__name__)

class Binarizator(AbstractWorker):
    
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.config = config
        self.geometric_bin = config.get('polygon_config', {})
        self.binarize_config = config.get('binarize', {})
        self.c_value = self.binarize_config.get('c_value', 7)
        self.height_thresholds = self.binarize_config.get('height_thresholds_px', [100, 800, 1500, 2500])
        self.block_sizes_map = self.binarize_config.get('block_sizes_map', [15, 21, 25, 35, 41])

        
    def _measure_polygon_quality(self, gray_img: np.ndarray[Any, Any]) -> str:
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
        
    def _otsu_binarize(self, gray_img: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Binarización Otsu."""
        _, bin_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return bin_img
       
    def _adaptive_binarize(self, gray_img: np.ndarray[Any, Any], block_size: int) -> np.ndarray[Any, Any]:
        """Binarización adaptativa Gaussiana."""
        return cv2.adaptiveThreshold(
            gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, block_size, self.c_value
        )
    
    def _sauvola_binarize(self, gray_img: np.ndarray[Any, Any], adaptive_block_size: int) -> Optional[np.ndarray[np.uint8, Any]]:
        """Binarización Sauvola de scikit-image."""
        thresh_sauvola: np.ndarray[Any, Any]
        thresh_sauvola = threshold_sauvola(gray_img, adaptive_block_size) # type: ignore
        if thresh_sauvola is None:
            raise ValueError("thresh_sauvola devolvío None")
        
        bin_image: np.ndarray = np.asarray(img_as_ubyte(gray_img, True), dtype=np.uint8)# type: ignore
        binary_result: np.ndarray = bin_image > thresh_sauvola# type: ignore
        bin_img: np.ndarray = (binary_result * 255).astype(np.uint8)# type: ignore
        return bin_img# type: ignore

    def _adaptive_mean_fallback(self, gray_img: np.ndarray[Any, Any], block_size: int) -> np.ndarray[Any, Any]:
        """Fallback con adaptive mean de OpenCV."""
        return cv2.adaptiveThreshold(
            gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, block_size, max(1, self.c_value - 2)
        )
   
    def _process_individual_polygons(self, gray_img: np.ndarray[Any, Any], height: float) -> np.ndarray[Any, Any]:
        """Binarización basada en una sola medición de calidad local."""
        adaptive_block_size = self._get_adaptive_block_size(height)
        metodo = self._measure_polygon_quality(gray_img)
        logger.debug(f"Selección de método '{metodo}' para polígono de altura {height}")

        if metodo == "otsu":
            bin_img = self._otsu_binarize(gray_img)
        elif metodo == "adaptive_gaussian":
            bin_img = self._adaptive_binarize(gray_img, adaptive_block_size)
        elif metodo == "sauvola":
            bin_img = self._sauvola_binarize(gray_img, adaptive_block_size)
        else:
            bin_img = self._adaptive_mean_fallback(gray_img, adaptive_block_size)

        return bin_img
    
    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """Procesa un diccionario de polígonos individuales aplicando binarización adaptativa a cada uno. Devuelve un diccionario con polygon_id -> binarized_img"""
        polygons = context.get("polygons", {})
        polygons_to_bin = polygons.copy()
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
                cropped_img: Optional[np.ndarray[Any, Any]]
                height = cropped_img.shape[0]
                gray_img: Optional[np.ndarray[Any, Any]] = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY) if len(cropped_img.shape) == 3 else cropped_img
                
                bin_img: Optional[np.ndarray[Any, Any]] = self._process_individual_polygons(gray_img, height)

                # Guardar solo el polygon_id y la imagen binarizada
                binarized_polygons[polygon_id] = bin_img
                binarized_count += 1

            except Exception as e:
                logger.error(f"Error procesando polígono {polygon_id}: {e}")
                failed_count += 1
        
        total_processed = len(polygons_to_bin)
        logger.info(f"Binarización completada: {binarized_count}/{total_processed} polígonos procesados.")