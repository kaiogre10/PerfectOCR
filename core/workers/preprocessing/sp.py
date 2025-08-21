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
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = self.config.get('median_filter', {})
        self.enabled_outputs = self.config.get("enabled_outputs", {})
        self.output = self.enabled_outputs.get("sp_poly", False)
    
    def preprocess(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """
        Detecta y corrige patrones de sal y pimienta en cada polígono del diccionario,
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
                            
            processed_img = self._detect_sp_single(cropped_img)
            
            cropped_img[...] = processed_img
            
            total_time = time.time() - start_time
            logger.debug(f"Moire completado en: {total_time:.3f}s")

            if self.output:
                from services.output_service import save_image
                import os
                
                output_paths = context.get("output_paths", [])
                poly_id = context.get("poly_id", "unknown_poly")
                
                for path in output_paths:
                    output_dir = os.path.join(path, "sp")
                    file_name = f"{poly_id}_sp_debug.png"
                    save_image(processed_img, output_dir, file_name)
                
                if output_paths:
                    logger.debug(f"Imagen de debug de S&P para '{poly_id}' guardada en {len(output_paths)} ubicaciones.")
            return True
        except Exception as e:
            logger.error(f"Error en manejo de  {e}")
            return False
    
    def _detect_sp_single(self, cropped_img: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Filtro adaptativo de sal y pimienta a nivel token (polígono), DPI-invariante y seguro para texto."""
        try:
            total_pixels = cropped_img.size
            if total_pixels == 0:
                return cropped_img

            h, w = cropped_img.shape[:2]
            area = h * w

            # Normalizar a uint8 solo si es necesario (para operar con OpenCV), luego reescalar de vuelta
            orig_dtype = cropped_img.dtype
            img_min = None
            img_max = None
            if orig_dtype != np.uint8:
                img_min = float(np.min(cropped_img))
                img_max = float(np.max(cropped_img))
                if img_max > img_min:
                    normalized = ((cropped_img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    normalized = cropped_img.astype(np.uint8)
            else:
                normalized = cropped_img

            # Percentiles locales: extremos adaptativos (DPI-invariante)
            p1 = float(np.percentile(normalized, 1))
            p5 = float(np.percentile(normalized, 5))
            p95 = float(np.percentile(normalized, 95))
            p99 = float(np.percentile(normalized, 99))
            contrast_range = p99 - p1

            if contrast_range > 150.0:
                low = int(max(0, p1))
                high = int(min(255, p99))
            else:
                low = int(max(0, p5))
                high = int(min(255, p95))

            # Máscara de extremos (candidatos a SP)
            extreme_mask = (normalized <= low) | (normalized >= high)
            sp_pixels = int(np.count_nonzero(extreme_mask))
            sp_ratio = sp_pixels / float(area)

            # Contar aislados: vecinos 8-conectados (impulsos sueltos)
            kernel = np.ones((3, 3), np.uint8)
            neighbor_count = cv2.filter2D(extreme_mask.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_REPLICATE)
            isolated_mask = extreme_mask & (neighbor_count <= 1)
            isolated_count = int(np.count_nonzero(isolated_mask))

            # Umbrales adaptativos por tamaño de token
            if area < 2000:
                ratio_thr, min_iso, ksize = 0.06, 10, 3
            elif area < 10000:
                ratio_thr, min_iso, ksize = 0.03, 20, 3
            else:
                ratio_thr, min_iso, ksize = 0.015, 30, 5

            if min(h, w) >= 50:
                ksize = max(ksize, 5)
            if ksize % 2 == 0:
                ksize += 1

            # Activación estricta: SP real (no bordes normales)
            if not (sp_ratio > ratio_thr and isolated_count >= min_iso):
                processed_img = cropped_img
                return processed_img
            # Salvaguarda de nitidez (Sobel) antes/después
            sobel_before = np.mean(np.abs(cv2.Sobel(normalized, cv2.CV_64F, 1, 1, ksize=3)))

            # Mediana + reemplazo selectivo SOLO sobre la máscara de extremos (preserva trazos)
            filtered = cv2.medianBlur(normalized, ksize)
            if filtered is None:
                processed_img = cropped_img
                return processed_img

            result_u8 = normalized.copy()
            result_u8[extreme_mask] = filtered[extreme_mask]

            sobel_after = np.mean(np.abs(cv2.Sobel(result_u8, cv2.CV_64F, 1, 1, ksize=3)))
            if sobel_after < 0.85 * sobel_before:
                
                processed_img = cropped_img
                return processed_img
                
            # Escribir in-place respetando dtype original
            if orig_dtype != np.uint8:
                if img_max is None or img_min is None or img_max <= img_min:
                    cropped_img[...] = result_u8.astype(orig_dtype)
                else:
                    # Aseguramos que img_max y img_min no sean None antes de operar
                    scale = float(img_max) - float(img_min)
                    back = (result_u8.astype(np.float32) / 255.0 * scale + float(img_min)).astype(orig_dtype)
                    cropped_img[...] = back
            else:
                cropped_img[...] = result_u8

            processed_img = cropped_img
            return processed_img

        except cv2.error as e:
            logger.warning(f"OpenCV en DoctorSaltPepper: {e}, manteniendo imagen original")
            processed_img = cropped_img
            return processed_img
        except Exception as e:
            logger.warning(f"SP adaptativo falló: {e}, manteniendo imagen original")
            processed_img = cropped_img
            return processed_img