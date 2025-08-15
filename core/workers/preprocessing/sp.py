# PerfectOCR/core/workflow/preprocessing/sp.py
import cv2
import numpy as np
import logging
import time
from typing import Dict, Any 
from core.factory.abstract_worker import PreprossesingAbstractWorker
from core.domain.data_formatter import DataFormatter
logger = logging.getLogger(__name__)
    
class DoctorSaltPepper(PreprossesingAbstractWorker):
    total_polygons_corrected = 0
    
    def __init__(self, config: Dict[str, Any], project_root: str):    
        self.project_root = project_root
        self.config = config
    
    def preprocess(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """
        Detecta y corrige patrones de sal y pimienta en cada polígono del diccionario,
        modificando 'cropped_img' in-situ.
        """
        try:
            start_time = time.time()
            cropped_image = context.get("cropped_img", {})

            cropped_img = np.array(cropped_image)
            if cropped_img.size == 0:
                error_msg = f"Imagen vacía o corrupta en '{cropped_img}'"
                logger.error(error_msg)
                context['error'] = error_msg
                return False
        
            if len(cropped_img.shape) == 3:
                cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                logger.info("convirtiendo a escala de grises")
            else:
                cropped_img = cropped_img
                    
            processed_img = self._detect_sp_single(cropped_img)
            
            cropped_img[...] = processed_img
            
            total_time = time.time() - start_time
            logger.debug(f"Moire completado en: {total_time:.3f}s")
            return True
        except Exception as e:
            logger.error(f"Error en manejo de  {e}")
            return False
    
    def _detect_sp_single(self, cropped_img: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Estima ruido sal y pimienta con histograma, operando siempre in-place sobre el array original."""
        sp_corrections = self.config.get('median_filter', {})
        low_thresh = int(sp_corrections.get('salt_pepper_low', 10))
        high_thresh = int(sp_corrections.get('salt_pepper_high', 245))
        kernel_size = int(sp_corrections.get('kernel_size', 3))
        sp_threshold = float(sp_corrections.get('salt_pepper_threshold', 0.001))

        total_pixels = cropped_img.size
        if total_pixels == 0:
            return cropped_img

        # Calcular histograma
        hist, _ = np.histogram(cropped_img, bins=256, range=(0, 255))

        # Sumar píxeles en los extremos del histograma
        sp_pixels = int(np.sum(hist[:low_thresh]) + np.sum(hist[high_thresh:]))
        sp_ratio = sp_pixels / total_pixels

        if sp_ratio > sp_threshold:
            # Ajustar kernel_size a impar si es necesario
            if kernel_size % 2 == 0:
                kernel_size += 1

            try:
                # Mover img_min e img_max aquí para que estén en scope
                img_min = np.min(cropped_img)
                img_max = np.max(cropped_img)
                
                if cropped_img.dtype != np.uint8:
                    if img_max > img_min:
                        normalized = ((cropped_img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                    else:
                        normalized = cropped_img.astype(np.uint8)
                else:
                    normalized = cropped_img
            
                # Aplicar filtro de mediana
                filtered = cv2.medianBlur(normalized, kernel_size)
                if cropped_img.dtype != np.uint8:
                    filtered_scaled = (filtered.astype(np.float32) / 255.0 * (img_max - img_min) + img_min).astype(cropped_img.dtype)
                    cropped_img[...] = filtered_scaled
                else:
                    cropped_img[...] = filtered
                
            except cv2.error as e:
                logger.warning(f"Polígono sin suficiente ruido: {e}")
                pass
        
        return cropped_img
        