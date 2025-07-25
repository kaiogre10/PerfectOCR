# PerfectOCR/core/workflow/geometry/binarization.py
from sklearnex import patch_sklearn
patch_sklearn
import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from skimage.filters import threshold_sauvola
from skimage.util import img_as_ubyte

logger = logging.getLogger(__name__)

class Binarizator:
    
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.geometric = config
        self.geometric_bin = config.get('polygon_config', {})
        
        # Configuraciones de binarización
        binarize_config = self.geometric_bin.get('binarize', {})
        self.c_value = binarize_config.get('c_value', 7)
        self.height_thresholds = binarize_config.get('height_thresholds_px', [100, 800, 1500, 2500])
        self.block_sizes_map = binarize_config.get('block_sizes_map', [15, 21, 25, 35, 41])
        self.quality_threshold = binarize_config.get('quality_threshold', [0.1, 0.9])
        self.quality_min, self.quality_max = self.quality_threshold
           
    def _check_quality(self, binary_img: np.ndarray) -> bool:
        """Verifica calidad basada en ratio de píxeles blancos."""
        if binary_img is None or binary_img.size == 0:
            return False
        white_ratio = np.sum(binary_img == 255) / binary_img.size
        return self.quality_min < white_ratio < self.quality_max

    def _is_histogram_bimodal(self, gray_img: np.ndarray) -> bool:
        """Estima si el histograma es bimodal, bueno para Otsu."""
        hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
        hist = hist.flatten()
        
        # Encontrar picos, ignorando los extremos (negro/blanco puro)
        peaks = []
        for i in range(1, 255):
            if hist[i-1] < hist[i] and hist[i+1] < hist[i]:
                peaks.append(hist[i])
        
        # Un histograma bimodal para texto suele tener 2 picos dominantes.
        return len(peaks) >= 2
       
    def _otsu_binarize(self, gray_img: np.ndarray) -> np.ndarray:
        """Binarización Otsu."""
        _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
   
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
    
    def _get_adaptive_block_size(self, height: float) -> int:
        """Determina el block_size adaptativo basado en la altura del polígono."""
        for i, threshold in enumerate(self.height_thresholds):
            if height <= threshold:
                block_size = self.block_sizes_map[min(i, len(self.block_sizes_map) - 1)]
                # Asegurar que sea impar y >= 3
                return max(3, block_size if block_size % 2 != 0 else block_size + 1)
        # Si es más alto que todos los thresholds, usar el block_size más grande
        final_block_size = self.block_sizes_map[-1]
        return max(3, final_block_size if final_block_size % 2 != 0 else final_block_size + 1)
   
    def _binarize_single_polygon(self, polygon_img: np.ndarray, polygon_height: float) -> Tuple[np.ndarray, str]:
        """
        Binarización robusta y adaptativa para un polígono individual.
        Prueba una cascada de métodos desde el más rápido al más robusto.
        """
        if polygon_img.size == 0:
            logger.warning("Polígono vacío recibido para binarización")
            return polygon_img, "empty"
            
        gray_img = cv2.cvtColor(polygon_img, cv2.COLOR_BGR2GRAY) if len(polygon_img.shape) == 3 else polygon_img.copy()
        
        adaptive_block_size = self._get_adaptive_block_size(polygon_height)
        logger.debug(f"Usando block_size adaptativo {adaptive_block_size} para polígono de altura {polygon_height}")

        # 1. Otsu: Rápido y bueno para imágenes de alto contraste.
        if self._is_histogram_bimodal(gray_img):
            try:
                otsu_result = self._otsu_binarize(gray_img)
                if self._check_quality(otsu_result):
                    logger.debug("Binarización exitosa con método Otsu")
                    return otsu_result, "otsu"
            except Exception as e:
                logger.warning(f"Error en binarización Otsu: {e}")
       
        # 2. Adaptive Gaussian (OpenCV): Bueno para iluminación variable.
        try:
            adaptive_result = self._adaptive_binarize(gray_img, adaptive_block_size)
            if self._check_quality(adaptive_result):
                logger.debug("Binarización exitosa con método Adaptive Gaussian")
                return adaptive_result, "adaptive_gaussian"
        except Exception as e:
            logger.warning(f"Error en binarización Adaptive Gaussian: {e}")

        # 3. Sauvola (scikit-image): "Estándar de oro" para casos difíciles.
        try:
            sauvola_result = self._sauvola_binarize(gray_img, adaptive_block_size)
            if self._check_quality(sauvola_result):
                logger.debug("Binarización exitosa con método Sauvola (skimage)")
                return sauvola_result, "sauvola"
        except Exception as e:
            logger.warning(f"Error en binarización Sauvola: {e}")
       
        # 4. Fallback final: Adaptive Mean (OpenCV).
        try:
            fallback_result = self._adaptive_mean_fallback(gray_img, adaptive_block_size)
            logger.debug("Usando método fallback Adaptive Mean")
            return fallback_result, "adaptive_mean_fallback"
        except Exception as e:
            logger.error(f"Error en método fallback final: {e}")
            return gray_img, "failed"

    def _process_individual_polygons(self, individual_polygons: List[Dict]) -> List[Dict]:
        """
        Procesa una lista de polígonos individuales aplicando binarización adaptativa a cada uno.
        Reemplaza la imagen original con la versión binarizada.
        Utiliza la metadata existente sin modificarla.
        
        Args:
            individual_polygons: Lista de polígonos con sus imágenes recortadas y metadata
            
        Returns:
            binarized_poly: Los polígonos con sus imágenes binarizadas
        """
        if not individual_polygons:
            logger.warning("Lista de polígonos vacía recibida")
            return []
        
        logger.info(f"Iniciando binarización de {len(individual_polygons)} polígonos individuales")
        
        binarized_poly = []
        binarized_count = 0
        failed_count = 0
        
        for idx, polygon in enumerate(individual_polygons):
            try:
                # Obtener la imagen recortada del polígono
                cropped_img = polygon.get("cropped_img")
                if cropped_img is None:
                    logger.warning(f"Polígono {polygon.get('polygon_id', idx)} sin imagen, omitiendo")
                    failed_count += 1
                    continue
                    
                # Usar la altura directamente de la metadata, sin valores por defecto
                height = polygon["height"] 
                
                # Binarizar la imagen y reemplazar la original
                binarized_poly, _ = self._binarize_single_polygon(cropped_img, height)
                
            except Exception as e:
                logger.error(f"Error procesando polígono {polygon.get('polygon_id', idx)}: {e}")
                failed_count += 1
                
        logger.info(f"Binarización completada: {binarized_count}/{len(individual_polygons)} polígonos procesados exitosamente")
        
        return binarized_poly
        
        # 1. Análisis de componentes y ruido
        # labeled_regions, num_regions = ndimage.label(binary)
        # if num_regions > 0:
        #     region_sizes = np.bincount(labeled_regions.ravel())[1:]
        #     small_regions = np.sum(region_sizes < 20)  # 20 píxeles como umbral mínimo
        #     noise_ratio = small_regions / (num_regions + 1e-8)
        #     
        #     # 2. Reducción de ruido adaptativa
        #     if noise_ratio > 0.3:
        #         binary = morphology.remove_small_objects(
        #             binary.astype(bool), 
        #             min_size=20
        #         ).astype(np.uint8) * 255
        #     
        #     # 3. Ajuste morfológico adaptativo
        #     kernel_size = max(2, int(min(gray.shape) * 0.005) // 2 * 2 + 1)
        #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        #     
        #     # Si hay muchos componentes pequeños, aplicar closing
        #     # Si hay componentes fragmentados, aplicar opening
        #     if noise_ratio > 0.2:
        #         binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        #     else:
        #         # Verificar fragmentación basada en estadísticas simples
        #         avg_size = np.mean(region_sizes)
        #         if avg_size < 100:  # Umbral arbitrario para texto fragmentado
        #             binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        #         
        # if abs(props["dominant_angle"]) > 2:
        #     # Corregir rotación si es significativa
        #     angle = props["dominant_angle"]
        #     binary = ndimage.rotate(binary, angle, reshape=False, mode='constant')