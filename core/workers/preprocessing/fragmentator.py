# PerfectOCR/core/workflow/preprocessing/fragmentator.py
import cv2
import logging
import numpy as np
from typing import Dict, Any, Set
from core.factory.abstract_worker import PreprossesingAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import CroppedImage

logger = logging.getLogger(__name__)

class PolygonFragmentator(PreprossesingAbstractWorker):
    
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.config = config
        fragmentator_params = self.config.get('fragmentation', {})
        self.min_area_factor = fragmentator_params.get('min_area_factor', 0.01)
        self.density_std_factor = fragmentator_params.get('density_std_factor', 1.0)
        self.approx_poly_epsilon = fragmentator_params.get('approx_poly_epsilon', 0.02)
        self.problematic_ids: Set[str] = set()
        self.binarized_images: Dict[str, np.ndarray[np.ndarray[np.uint8, Any]]] = {}
        
    def set_binarized_images(self, binarized_images: Dict[str, np.ndarray[np.uint8, Any]]):
        """Recibe las imágenes binarizadas del stager."""
        self.binarized_images = binarized_images
        logger.info(f"Fragmentador recibió {len(binarized_images)} imágenes binarizadas")

    def preprocess(self, cropped_img: CroppedImage, manager: DataFormatter) -> CroppedImage:
        """Analiza y fragmenta el polígono si es necesario, manteniendo la reindexación."""
        try:
            poly_id = cropped_img.polygon_id
            
            # Verificar si tenemos imágenes binarizadas
            binarized_images: Dict[str, np.ndarray[np.uint8, Any]]
            
            bin_img = self.binarized_images.get(poly_id)
            bin_img = np.ndarray[np.uint8, Any]
            if bin_img is None:
                logger.debug(f"No hay imagen binarizada para {poly_id}")
                return cropped_img
            
            # Analizar si necesita fragmentación
            if self._needs_fragmentation(bin_img):
                logger.info(f"Polígono {poly_id} necesita fragmentación")
                
                # Fragmentar la imagen original
                fragmented_img = self._fragment_image(cropped_img.cropped_img, bin_img)
                
                cropped_img.cropped_img[...] = fragmented_img


            else:
                logger.debug(f"Polígono {poly_id} no necesita fragmentación")
                return cropped_img
                
        except Exception as e:
            logger.error(f"Error en fragmentación del polígono {cropped_img.poly_id}: {e}")
            return cropped_img

    def _needs_fragmentation(self, bin_img: np.ndarray[np.uint8, Any]) -> bool:
        """Determina si la imagen binarizada necesita fragmentación."""
    
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False

        # Calcula perímetros de todos los contornos
        perimeters = [cv2.arcLength(c, True) for c in contours]
        mean_perim = np.mean(perimeters)
        std_perim = np.std(perimeters)

        # Filtra contornos cuyo perímetro esté dentro de 2 desviaciones estándar del promedio
        valid_contours = [
            c for c, p in zip(contours, perimeters)
            if abs(p - mean_perim) <= 2 * std_perim
        ]

        num_valid = len(valid_contours)
        
        # Considera problemático si hay muchos contornos (agrupamiento incorrecto)
        if num_valid > 3:  # Umbral ajustable
            return True
            
        # También verifica el área de los contornos
        poly_area = bin_img.shape[0] * bin_img.shape[1]
        adaptive_min_area = poly_area * self.min_area_factor
        
        large_contours = [c for c in valid_contours if cv2.contourArea(c) > adaptive_min_area]
        
        if len(large_contours) > 1:
            return True
            
        return False
        
    def _fragment_image(self, cropped_img: np.ndarray[Any, Any], bin_img: np.ndarray[np.uint8, Any]) -> np.ndarray[Any, Any]:
        """Fragmenta la imagen original basándose en el análisis de la binarizada."""
        try:
            # Encontrar contornos en la imagen binarizada
            contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return cropped_img
            
            # Filtrar contornos por área mínima
            poly_area = cropped_img.shape[0] * cropped_img.shape[1]
            adaptive_min_area = poly_area * self.min_area_factor
            valid_contours = [c for c in contours if cv2.arcLength(c, True) > adaptive_min_area]
            
            if len(valid_contours) <= 1:
                return cropped_img
            
            # Ordenar contornos por posición X (izquierda a derecha)
            sorted_contours = sorted(valid_contours, key=lambda c: cv2.boundingRect(c)[0])
            
            # Crear máscara combinada de todos los fragmentos
            fragment_mask = np.zeros_like(bin_img)
            for contour in sorted_contours:
                cv2.fillPoly(fragment_mask, [contour], 255)
            
            # Aplicar la máscara a la imagen original
            fragmented_img = cv2.bitwise_and(cropped_img, fragment_mask)
            
            logger.info(f"Imagen fragmentada en {len(sorted_contours)} fragmentos")
            return fragmented_img
            
        except Exception as e:
            logger.error(f"Error fragmentando imagen: {e}")
            return cropped_img

    def _get_problematic_ids(self) -> Set[str]:
        """Retorna los IDs de polígonos problemáticos del último procesamiento."""
        return self.problematic_ids
