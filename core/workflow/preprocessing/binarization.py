# PerfectOCR/core/workflow/preprocessing/binarization.py
import cv2
import numpy as np
import logging
from scipy import ndimage
from skimage import filters, morphology 
from typing import Dict, Any

logger = logging.getLogger(__name__)

class Binarizator:
    
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.corrections = config
            
    def _estimate_binarization(self, image_to_binarize: np.ndarray) -> np.ndarray:
        binarize_corrections = self.corrections
        window = binarize_corrections.get('window_size', 25)
        k = binarize_corrections.get('k', 0.2)
        adaptive_c = binarize_corrections.get('adaptive_c_value', 7)
        height_thresholds = binarize_corrections.get('height_thresholds_px', [800, 1500, 2500])
        block_size = binarize_corrections.get('block_sizes_map', [21, 25, 35, 41])
        
        # Normalización del histograma para mejorar el contraste global
        gray_eq = cv2.equalizeHist(image_to_binarize)
        
        # Evaluación de la complejidad de la imagen
        gradient_magnitude = cv2.Sobel(gray_eq, cv2.CV_64F, 1, 1)
        gradient_mean = np.mean(np.abs(gradient_magnitude))
        gradient_std = np.std(np.abs(gradient_magnitude))
        complexity_score = gradient_std / (gradient_mean + 1e-8)
        
        # Selección del método de binarización según complejidad
        if complexity_score > 1.5:
            # Imagen compleja con variaciones de iluminación
            # Usar binarización adaptativa con bloques pequeños
            block_size = max(11, int(min(image_to_binarize.shape) * 0.02) // 2 * 2 + 1)
            binarized_img = cv2.adaptiveThreshold(
                gray_eq, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 
                block_size, 
                2
            )
        else:
            # Imagen más uniforme
            # Usar método de Otsu con ajuste por regiones
            global_thresh = filters.threshold_otsu(gray_eq)
            
            # Dividir la imagen en regiones para binarización local
            region_height = max(image_to_binarize.shape[0] // 4, 1)
            region_width = max(image_to_binarize.shape[1] // 4, 1)
            binarized_img = np.zeros_like(gray_eq, dtype=np.uint8)
            
            for y in range(0, image_to_binarize.shape[0], region_height):
                for x in range(0, image_to_binarize.shape[1], region_width):
                    # Extraer región
                    region = gray_eq[y:min(y+region_height, image_to_binarize.shape[0]), 
                                     x:min(x+region_width, image_to_binarize.shape[1])]
                    
                    # Calcular umbral local (con fallback a global)
                    try:
                        # Intentar umbral de Otsu local
                        local_thresh = filters.threshold_otsu(region)
                        # Si el umbral local se desvía demasiado, usar global
                        if abs(local_thresh - global_thresh) > 50:
                            thresh = (local_thresh + global_thresh) / 2
                        else:
                            thresh = local_thresh
                    except:
                        # Fallback a umbral global
                        thresh = global_thresh
                    
                    # Aplicar umbral a la región
                    region_binary = (region < thresh).astype(np.uint8) * 255
                    binarized_img[y:min(y+region_height, binarized_img.shape[0]), 
                           x:min(x+region_width, binarized_img.shape[1])] = region_binary
                else:
                    binarized_img = image_to_binarize
        
        return binarized_img
    
    
        



        # Post-procesamiento para mejorar calidad
        
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