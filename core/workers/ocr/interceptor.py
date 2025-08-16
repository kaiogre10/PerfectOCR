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
        self.min_area_factor = frag_config.get('min_area_factor', 0.01)
        self.min_contours_for_frag = frag_config.get('min_contours_for_frag', 3)
        self.approx_poly_epsilon = frag_config.get('approx_poly_epsilon', 0.02)
        

    def intercept_polygons(
        self,
        polygon_ids: List[str],
        cleaned_batch_results: List[Optional[Dict[str, Any]]],
        fragmentation_candidates: List[Tuple[str, int]],  # (poly_id, fragmentos_necesarios)
        image_list_copy: List[np.ndarray[Any, Any]],
        polygons: Dict[str, Any]
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Divide polígonos basándose en la sugerencia del TextCleaner.
        Mantiene la estructura original de PaddleOCR con reindexado correcto.
        """
        # Si no hay sugerencias, devolver los resultados originales
        if not fragmentation_candidates:
            return cleaned_batch_results

        # Crear diccionario de imágenes por ID
        images_by_id = dict(zip(polygon_ids, image_list_copy))
        
        # Lista final de resultados
        batch_results = []
        next_polygon_id = len(polygon_ids)  # ID para el siguiente fragmento
        
        # Procesar cada resultado en orden
        for i, (poly_id, result) in enumerate(zip(polygon_ids, cleaned_batch_results)):
            # Verificar si este polígono necesita fragmentación
            needs_fragmentation = any(sug[0] == poly_id for sug in fragmentation_candidates)
            
            if needs_fragmentation:
                # Obtener el número de fragmentos necesarios
                num_fragmentos = next(sug[1] for sug in fragmentation_candidates if sug[0] == poly_id)
                
                # Obtener la imagen del polígono
                if poly_id in images_by_id:
                    img = images_by_id[poly_id]
                    if img is not None and img.size > 0:
                        # Binarizar la imagen para encontrar contornos
                        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) > 2 else img
                        bin_img = self._process_individual_polygon(gray_img)
                        
                        # Encontrar contornos en la imagen binarizada
                        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if len(contours) >= num_fragmentos:
                            # Filtrar contornos muy pequeños
                            poly_area = bin_img.shape[0] * bin_img.shape[1]
                            min_area = poly_area * self.min_area_factor
                            valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
                            
                            if len(valid_contours) >= num_fragmentos:
                                # Ordenar contornos de izquierda a derecha, de arriba a abajo
                                sorted_contours = sorted(valid_contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))
                                contours_to_use = sorted_contours[:num_fragmentos]
                                
                                # Obtener geometría del polígono original
                                original_polygon = polygons.get(poly_id)
                                if original_polygon:
                                    offset_x, offset_y, _, _ = original_polygon.get('geometry', {}).get('bounding_box', [0, 0, 0, 0])
                                    
                                    logger.info(f"Fragmentando polígono '{poly_id}' en {len(contours_to_use)} fragmentos.")
                                    
                                    # Para cada contorno (fragmento):
                                    for j, contour in enumerate(contours_to_use):
                                        # Simplificar la geometría del contorno
                                        epsilon = self.approx_poly_epsilon * cv2.arcLength(contour, True)
                                        approx_poly = cv2.approxPolyDP(contour, epsilon, True)
                                        
                                        # Traducir las coordenadas del contorno (locales al recorte) a
                                        # coordenadas absolutas (globales en la imagen principal).
                                        absolute_points = [[int(point[0][0] + offset_x), int(point[0][1] + offset_y)] for point in approx_poly]

                                        if j == 0:
                                            # Primer fragmento: mantiene ID original
                                            fragment_result = result.copy() if result else {}
                                            fragment_result.update({
                                                "polygon_id": poly_id,           # MISMO ID (ej: "poly_0004")
                                                "polygon_points": absolute_points,
                                            })
                                            batch_results.append(fragment_result)
                                        else:
                                            # Fragmentos adicionales: IDs secuenciales nuevos
                                            additional_fragment = {
                                                "polygon_id": f"poly_{next_polygon_id:04d}",  # NUEVO ID secuencial
                                                "polygon_points": absolute_points,
                                                "text": "",
                                                "confidence": 0.0,
                                                # Se mantienen otros campos del original
                                            }
                                            batch_results.append(additional_fragment)
                                            next_polygon_id += 1
                                    
                                    # Continuar con el siguiente polígono
                                    continue
                    
                # Si no se pudo fragmentar, mantener el resultado original
                batch_results.append(result)
            else:
                # Este polígono no fue fragmentado, mantener el resultado original
                batch_results.append(result)
        
        return batch_results

    def _process_individual_polygon(self, gray_img: np.ndarray[Any, Any]) -> np.ndarray[np.uint8, Any]:
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

    def _visual_analysis_needs_fragmentation(self, bin_img: np.ndarray[np.uint8, Any]) -> Tuple[bool, List[Any]]:
        """Determina si la imagen binarizada parece contener múltiples elementos separados.
        Devuelve un bool indicando necesidad de fragmentar y la lista de contornos válidos."""
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, []

        # Filtrar contornos muy pequeños que son probablemente ruido
        poly_area = bin_img.shape[0] * bin_img.shape[1]
        min_area = poly_area * self.min_area_factor
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

        return (len(valid_contours) >= self.min_contours_for_frag), valid_contours

    def _measure_polygon_quality(self, gray_img: np.ndarray[Any, Any]) -> str:
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
        
    def _otsu_binarize(self, gray_img: np.ndarray[Any, Any]) -> np.ndarray[np.uint8, Any]:
        _, bin_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return bin_img
       
    def _adaptive_binarize(self, gray_img: np.ndarray[Any, Any], block_size: int) -> np.ndarray[np.uint8, Any]:
        return cv2.adaptiveThreshold(
            gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, block_size, self.c_value
        )
    
    def _sauvola_binarize(self, gray_img: np.ndarray[Any, Any], adaptive_block_size: int) -> np.ndarray[np.uint8, Any]:
        thresh_sauvola = threshold_sauvola(gray_img, window_size=adaptive_block_size)
        bin_img = (gray_img > thresh_sauvola).astype(np.uint8) * 255
        return bin_img

    def _adaptive_mean_fallback(self, gray_img: np.ndarray[Any, Any], block_size: int) -> np.ndarray[np.uint8, Any]:
        return cv2.adaptiveThreshold(
            gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, block_size, max(1, self.c_value - 2)
        )
