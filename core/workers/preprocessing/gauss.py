# PerfectOCR/core/workflow/preprocessing/
import cv2
import numpy as np
import logging
from typing import Dict, Any
from core.factory.abstract_worker import PreprossesingAbstractWorker
from core.domain.data_formatter import DataFormatter
import time

logger = logging.getLogger(__name__)

class GaussianDenoiser(PreprossesingAbstractWorker):

    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = self.config.get('bilateral_params', {})
        self.enabled_outputs = self.config.get("enabled_outputs", {})
        self.output = self.enabled_outputs.get("gauss_poly", False)
        
    def preprocess(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """
        Detecta y corrige patrones de moiré en cada polígono del diccionario,
        modificando 'cropped_img' in-situ.
        """
        try:
            start_time = time.time()
            cropped_img = context.get("cropped_img")

            cropped_img = np.array(cropped_img)
            if cropped_img.size == 0:
                error_msg = f"Imagen vacía o corrupta en '{cropped_img}'"
                logger.error(error_msg)
                context['error'] = error_msg
                return False
                    
            processed_img = self._estimate_gaussian_noise_single(cropped_img)
            
            cropped_img[...] = processed_img
            
            total_time = time.time() - start_time
            logger.debug(f"Moire completado en: {total_time:.3f}s")

            if self.output:
                from services.output_service import save_image
                import os
                
                output_paths = context.get("output_paths", [])
                poly_id = context.get("poly_id", "unknown_poly")
                
                for path in output_paths:
                    output_dir = os.path.join(path, "gauss")
                    file_name = f"{poly_id}_gauss_debug.png"
                    save_image(processed_img, output_dir, file_name)
                
                if output_paths:
                    logger.debug(f"Imagen de debug Gauss para '{poly_id}' guardada en {len(output_paths)} ubicaciones.")
            return True
        except Exception as e:
            logger.error(f"Error en manejo de  {e}")
            return False

    def _estimate_gaussian_noise_single(self, cropped_img: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Estima ruido general con varianza del Laplaciano."""
        bilateral_corrections = self.config.get('bilateral_params', {})
        d = bilateral_corrections.get('d', 9)
        sigma_color = bilateral_corrections.get('sigma_color', 75)
        sigma_space = bilateral_corrections.get('sigma_space', 75)
        gauss_threshold = bilateral_corrections.get('laplacian_variance_threshold', 100)

        try:
            # Convertir a formato OpenCV compatible para Laplacian
            if cropped_img.dtype != np.uint8:
                img_min = np.min(cropped_img)
                img_max = np.max(cropped_img)
                if img_max > img_min:
                    normalized = ((cropped_img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    normalized = cropped_img.astype(np.uint8)
            else:
                normalized = cropped_img
            
            # Calcular varianza del Laplaciano
            laplacian_var = cv2.Laplacian(normalized, cv2.CV_64F).var()
            
            if laplacian_var < gauss_threshold:
                # Aplicar filtro bilateral
                processed_img = cv2.bilateralFilter(normalized, d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
                
                # Convertir de vuelta al formato original si es necesario
                if cropped_img.dtype != np.uint8:
                    processed_img = (processed_img.astype(np.float32) / 255.0 * (img_max - img_min) + img_min).astype(cropped_img.dtype)
            else:
                processed_img = cropped_img
                
            return processed_img
            
        except cv2.error as e:
            logger.warning(f"OpenCV falló en GaussianDenoiser: {e}, manteniendo imagen original")
            processed_img = cropped_img
            return processed_img