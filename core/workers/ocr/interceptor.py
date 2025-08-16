# PerfectOCR/core/workers/ocr/interceptor.py
import cv2
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple, Set
from skimage.filters import threshold_sauvola
from skimage.util import img_as_ubyte

logger = logging.getLogger(__name__)

class Interceptor:
    """
    Actúa como un centro de decisión para la fragmentación de polígonos.
    - Binariza las imágenes de los polígonos para su análisis.
    - Realiza un análisis visual para detectar agrupaciones incorrectas.
    - Combina su análisis visual con las sugerencias basadas en texto (del TextCleaner)
      para crear una lista definitiva de polígonos que necesitan ser fragmentados.
    """
    
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.config = config
        
        bin_config = self.config.get('binarize', {})
        self.c_value = bin_config.get('c_value', 7)
        self.height_thresholds = bin_config.get('height_thresholds_px', [100, 800, 1500, 2500])
        self.block_sizes_map = bin_config.get('block_sizes_map', [15, 21, 25, 35, 41])

        frag_config = self.config.get('fragmentation', {})
        self.min_area_factor = frag_config.get('min_area_factor', 0.001)
        self.min_contours_for_frag = frag_config.get('min_contours_for_frag', 3)
        
        frag_params = self.config.get('fragmentation', {})
        self.min_area_factor = frag_params.get('min_area_factor', 0.01)
        self.approx_poly_epsilon = frag_params.get('approx_poly_epsilon', 0.02)


    def _intercept_polygons(
        self,
        polygon_images: Dict[str, np.ndarray],
        text_based_candidates: List[str]
    ) -> Tuple[List[str], Dict[str, np.ndarray]]:
        """
        Decide qué polígonos necesitan fragmentación basándose en análisis de texto y visual.

        Args:
            polygon_images: Diccionario de {poly_id: image_np_array}.
            text_based_candidates: Lista de poly_ids sugeridos por TextCleaner.

        Returns:
            - Una lista final de IDs de polígonos que deben ser fragmentados.
            - Un diccionario de {poly_id: binarized_image} para los polígonos a fragmentar,
              que será usado por el Fragmentator.
        """
        final_fragmentation_ids: Set[str] = set(text_based_candidates)
        binarized_images_for_fragmenter: Dict[str, np.ndarray] = {}

        for poly_id, img in polygon_images.items():
            if img is None or img.size == 0:
                continue

            # 1. Binarizar la imagen para el análisis
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) > 2 else img
            bin_img = self._process_individual_polygon(gray_img)

            # 2. Realizar análisis visual para ver si necesita fragmentación
            if self._visual_analysis_needs_fragmentation(bin_img):
                logger.debug(f"Análisis visual sugiere fragmentar el polígono '{poly_id}'.")
                final_fragmentation_ids.add(poly_id)
            
            # 3. Si el polígono es un candidato final, guardar su imagen binarizada
            if poly_id in final_fragmentation_ids:
                binarized_images_for_fragmenter[poly_id] = bin_img
        
        logger.info(f"Decisión de intercepción: {len(final_fragmentation_ids)} polígonos marcados para fragmentar.")
        return list(final_fragmentation_ids), binarized_images_for_fragmenter

    def _process_individual_polygon(self, gray_img: np.ndarray) -> np.ndarray:
        """Binarización inteligente basada en la calidad local de la imagen."""
        height = gray_img.shape[0]
        adaptive_block_size = self._get_adaptive_block_size(height)
        method = self._measure_polygon_quality(gray_img)

        if method == "otsu":
            return self._otsu_binarize(gray_img)
        elif method == "adaptive_gaussian":
            return self._adaptive_binarize(gray_img, adaptive_block_size)
        elif method == "sauvola":
            return self._sauvola_binarize(gray_img, adaptive_block_size)
        else: # "adaptive_mean"
            return self._adaptive_mean_fallback(gray_img, adaptive_block_size)

    def _visual_analysis_needs_fragmentation(self, bin_img: np.ndarray) -> bool:
        """Determina si la imagen binarizada parece contener múltiples elementos separados."""
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False

        # Filtrar contornos muy pequeños que son probablemente ruido
        poly_area = bin_img.shape[0] * bin_img.shape[1]
        min_area = poly_area * self.min_area_factor
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        # Si después de filtrar el ruido quedan múltiples contornos, es un candidato
        return len(valid_contours) >= self.min_contours_for_frag

    # --- Métodos de Binarización (sin cambios) ---

    def _measure_polygon_quality(self, gray_img: np.ndarray) -> str:
        std = np.std(gray_img)
        if std == 0: return "adaptive_mean" # Imagen plana
        
        hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256]).flatten()
        peaks = np.sum((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:]))
        prob = hist / np.sum(hist)
        entropy = -np.sum(prob * np.log2(prob + 1e-8))

        if peaks >= 2 and std > 30: return "otsu"
        elif std > 20 and entropy > 4.5: return "adaptive_gaussian"
        elif std > 10: return "sauvola"
        else: return "adaptive_mean"

    def _get_adaptive_block_size(self, height: float) -> int:
        for i, threshold in enumerate(self.height_thresholds):
            if height <= threshold:
                block_size = self.block_sizes_map[min(i, len(self.block_sizes_map) - 1)]
                return max(3, block_size if block_size % 2 != 0 else block_size + 1)
        final_block_size = self.block_sizes_map[-1]
        return max(3, final_block_size if final_block_size % 2 != 0 else final_block_size + 1)
        
    def _otsu_binarize(self, gray_img: np.ndarray) -> np.ndarray:
        _, bin_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return bin_img
       
    def _adaptive_binarize(self, gray_img: np.ndarray, block_size: int) -> np.ndarray:
        return cv2.adaptiveThreshold(
            gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, block_size, self.c_value
        )
    
    def _sauvola_binarize(self, gray_img: np.ndarray, adaptive_block_size: int) -> np.ndarray:
        thresh_sauvola = threshold_sauvola(gray_img, window_size=adaptive_block_size)
        bin_img = (gray_img > thresh_sauvola).astype(np.uint8) * 255
        return bin_img

    def _adaptive_mean_fallback(self, gray_img: np.ndarray, block_size: int) -> np.ndarray:
        return cv2.adaptiveThreshold(
            gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, block_size, max(1, self.c_value - 2)
        )

    def calculate_new_geometries(
        self,
        manager: 'DataFormatter',
        ids_to_fragment: List[str],
        binarized_images: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
        """
        Analiza las imágenes binarizadas de los polígonos a fragmentar,
        encuentra los contornos internos y calcula sus geometrías absolutas.

        Args:
            manager: El DataFormatter para acceder a los datos de los polígonos originales.
            ids_to_fragment: Lista de IDs de polígonos a procesar.
            binarized_images: Dict con las imágenes binarizadas de esos polígonos.

        Returns:
            - Un diccionario con los datos de los nuevos polígonos a crear.
            - Una lista con los IDs de los polígonos originales que han sido reemplazados.
        """
        new_polygons_data: Dict[str, Dict[str, Any]] = {}
        processed_parent_ids: List[str] = []

        for poly_id in ids_to_fragment:
            original_polygon = manager.get_polygon_by_id(poly_id)
            bin_img = binarized_images.get(poly_id)

            if not original_polygon or bin_img is None:
                logger.warning(f"No se pudo fragmentar '{poly_id}', faltan datos originales o imagen binarizada.")
                continue

            # El offset es la esquina superior izquierda del bounding box del polígono original.
            # Esto es crucial para convertir las coordenadas de los fragmentos (locales)
            # a coordenadas absolutas en la imagen completa.
            offset_x, offset_y, _, _ = original_polygon.get('bounding_box', [0, 0, 0, 0])

            # Encontrar contornos en la imagen binarizada
            contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) <= 1:
                logger.debug(f"Polígono '{poly_id}' no fue fragmentado, no se encontraron suficientes contornos.")
                continue

            # Filtrar y ordenar contornos
            min_area = bin_img.shape[0] * bin_img.shape[1] * self.min_area_factor
            valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            
            # Ordenar de izquierda a derecha, de arriba a abajo para IDs predecibles
            sorted_contours = sorted(valid_contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))

            if len(sorted_contours) <= 1:
                continue

            logger.info(f"Fragmentando polígono '{poly_id}' en {len(sorted_contours)} nuevos polígonos.")

            for i, contour in enumerate(sorted_contours):
                new_poly_id = f"{poly_id}_frag{i}"
                
                # Simplificar la geometría del contorno
                epsilon = self.approx_poly_epsilon * cv2.arcLength(contour, True)
                approx_poly = cv2.approxPolyDP(contour, epsilon, True)

                # Traducir las coordenadas del contorno (locales al recorte) a
                # coordenadas absolutas (globales en la imagen principal).
                absolute_points = [[int(point[0][0] + offset_x), int(point[0][1] + offset_y)] for point in approx_poly]

                # Crear la estructura de datos para el nuevo polígono
                new_polygons_data[new_poly_id] = {
                    "polygon_id": new_poly_id,
                    "polygon_points": absolute_points,
                    "text": "",  # Se llenará en una futura pasada de OCR
                    "confidence": 0.0,
                    "parent_id": poly_id,
                    "status": "pending_ocr" # Estado para indicar que necesita ser procesado
                }
            
            processed_parent_ids.append(poly_id)

        return new_polygons_data, processed_parent_ids
