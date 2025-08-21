# PerfectOCR/core/workflow/preprocessing/sharp.py
import cv2
import time
import numpy as np
import logging
from typing import Dict, Any
from skimage.filters import unsharp_mask # type: ignore
from core.factory.abstract_worker import PreprossesingAbstractWorker
from core.domain.data_formatter import DataFormatter

logger = logging.getLogger(__name__)

class SharpeningEnhancer(PreprossesingAbstractWorker):

    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = self.config.get('sharpening', {})
        self.enabled_outputs = self.config.get("enabled_outputs", {})
        self.output = self.enabled_outputs.get("sharp_poly", False)

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
                    
            processed_img = self._estimate_sharpness_single(cropped_img)
            
            cropped_img[...] = processed_img
            
            total_time = time.time() - start_time
            logger.debug(f"Sharpening completado en: {total_time:.3f}s")

            if self.output:
                from services.output_service import save_image
                import os
                
                output_paths = context.get("output_paths", [])
                poly_id = context.get("poly_id", "unknown_poly")
                
                for path in output_paths:
                    output_dir = os.path.join(path, "sharp")
                    file_name = f"{poly_id}_sharp_debug.png"
                    save_image(processed_img, output_dir, file_name)
                
                if output_paths:
                    logger.debug(f"Imagen de debug de sharp para '{poly_id}' guardada en {len(output_paths)} ubicaciones.")
            return True
        except Exception as e:
            logger.error(f"Error en manejo de  {e}")
            return False

    def _estimate_sharpness_single(self, cropped_img: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Estima nitidez con Sobel."""
        sharpen_corrections = self.config.get('sharpening', {})
        radius = sharpen_corrections.get('radius', 1.0)
        amount = sharpen_corrections.get('amount', 1.5)

        try:
            # Convertir a formato OpenCV compatible para Sobel
            if cropped_img.dtype != np.uint8:
                img_min = np.min(cropped_img)
                img_max = np.max(cropped_img)
                if img_max > img_min:
                    normalized = ((cropped_img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    normalized = cropped_img.astype(np.uint8)
            else:
                normalized = cropped_img

            # Calcular Sobel sobre la imagen normalizada
            sobel = cv2.Sobel(normalized, cv2.CV_64F, 1, 1, ksize=3)
            sharpness = np.mean(np.abs(sobel))

            global_sharp_var = np.var(normalized)
            adaptative_sharp_threshold = max(30, global_sharp_var * 0.5)
            
            if sharpness < adaptative_sharp_threshold:
                radius = min(2.0, max(0.5, global_sharp_var - 0.02))
                amount = min(2.0, max(1.0, global_sharp_var - 0.03))

                sharpened: np.ndarray[np.uint8, Any] = unsharp_mask(normalized, radius=float(radius), amount=float(amount))

                if sharpened is not None:
                    processed_img = (sharpened * 255).astype(np.uint8)
                    
                    # Convertir de vuelta al formato original si es necesario
                    if cropped_img.dtype != np.uint8:
                            return (processed_img.astype(np.float32) / 255.0 * (img_max - img_min) + img_min).astype(cropped_img.dtype)
                    else:
                        processed_img = cropped_img
                
                    return processed_img
            
        except Exception as e:
            logger.warning(f"OpenCV Sobel falló: {e}, manteniendo imagen original")