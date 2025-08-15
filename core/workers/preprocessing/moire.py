# PerfectOCR/core/workflow/preprocessing/moire.py
import cv2
import numpy as np
import logging
import time
from typing import Dict, Any, Tuple
from core.factory.abstract_worker import PreprossesingAbstractWorker
from core.domain.data_formatter import DataFormatter

logger = logging.getLogger(__name__)

class MoireDenoiser(PreprossesingAbstractWorker):
    """Detecta y corrige patrones de moiré."""
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = self.config.get('moire', {})
        self.enabled_outputs = self.config.get("enabled_outputs", {})
        self.output = self.enabled_outputs.get("moire_poly", False)

    def preprocess(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """
        Detecta y corrige patrones de moiré en cada polígono del diccionario,
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
            
            h, w = cropped_img.shape[:2]
            img_dims: Tuple[int, int] = h, w

            if len(cropped_img.shape) == 3:
                cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
            else:
                cropped_img = cropped_img
                    
            processed_img = self._detect_moire_single(cropped_img, img_dims)
            
            cropped_img[...] = processed_img
            
            total_time = time.time() - start_time
            logger.debug(f"Moire completado en: {total_time:.3f}s")

            if self.output:
                from services.output_service import save_image
                import os
                
                output_paths = context.get("output_paths", [])
                poly_id = context.get("poly_id", "unknown_poly")
                
                for path in output_paths:
                    output_dir = os.path.join(path, "moire")
                    file_name = f"{poly_id}_moire_debug.png"
                    save_image(processed_img, output_dir, file_name)
                
                if output_paths:
                    logger.info(f"Imagen de debug de moiré para '{poly_id}' guardada en {len(output_paths)} ubicaciones.")

            return True
        except Exception as e:
            logger.error(f"Error en manejo de  {e}")
            return False
        
        
    def _detect_moire_single(self, cropped_img: np.ndarray[Any, Any], img_dims: Tuple[int , int]) -> np.ndarray[Any, Any]:
        moire_corrections = self.config
        mode = moire_corrections.get('mode', {})
        percentile_corrections = mode.get('percentile', {})
        notch_radius = int(percentile_corrections.get('notch_radius', 2))
        min_dist = int(percentile_corrections.get('min_distance_from_center', 200))
        
        h, w = img_dims
                                
        max_dim = max(h, w)
        spectrum_var = np.var(cropped_img)
        logger.debug(f"Varianza del espectro: {spectrum_var:.2f}")
        adaptive_notch = max(2, min(6, int(notch_radius * (spectrum_var / 1000.0) * (max_dim / 1000.0))))
        adaptive_min_dist = max(50, min(300, int(min_dist * (max_dim / 2000.0))))

        img_float = np.float32(cropped_img)
        f_transform = cv2.dft(np.asarray(img_float), flags=cv2.DFT_COMPLEX_OUTPUT)
        f_shifted = np.fft.fftshift(f_transform)
        
        magnitude_spectrum = cv2.magnitude(f_shifted[:,:,0], f_shifted[:,:,1])
        magnitude_spectrum = 20 * np.log(magnitude_spectrum + 1)
                
        temp_spectrum = magnitude_spectrum.copy()
        center_point = (w // 2, h // 2)
        cv2.circle(temp_spectrum, center_point, adaptive_min_dist, (0.0,), -1)

        valid_spectrum = temp_spectrum[temp_spectrum > 0]
        mean_energy = np.mean(valid_spectrum)
        std_energy = np.std(valid_spectrum)
        skewness = np.mean((valid_spectrum - mean_energy) ** 3) / (std_energy ** 3) if std_energy > 0 else 0

        logger.debug(f"mean_energy: {mean_energy:.2f}, std_energy: {std_energy:.2f}, skewness: {skewness:.2f}")

        if std_energy / mean_energy > 0.5 and skewness > 1.0:  # Alta variabilidad y asimetría
            adaptive_threshold = np.percentile(valid_spectrum, 98)
            adaptive_threshold = max(1000, min(5000000, adaptive_threshold * (spectrum_var / 100.0)))
            correction_mode = "percentile"
        elif std_energy / mean_energy > 0.3 and skewness < 0.5:  # Variabilidad moderada, distribución uniforme
            adaptive_threshold = mean_energy * 4.0
            adaptive_threshold = max(1000, min(5000000, adaptive_threshold * (spectrum_var / 50.0)))
            correction_mode = "factor"
        else:  # Baja variabilidad o picos fuertes
            adaptive_threshold = 2000000.0
            adaptive_threshold = max(1000, min(5000000, adaptive_threshold * (spectrum_var / 100.0)))
            correction_mode = "absolute"

        logger.debug(f"Modo de corrección aplicado: {correction_mode}, threshold: {adaptive_threshold:.2f}")

        peaks_coords = np.argwhere(magnitude_spectrum > adaptive_threshold)
        filtered_peaks = [(y, x) for y, x in peaks_coords if np.sqrt((y - h//2)**2 + (x - w//2)**2) > adaptive_min_dist]

        if len(filtered_peaks) > 0:
            logger.debug(f"Se detectaron {len(filtered_peaks)} picos fuera del centro. Aplicando corrección de moiré.")
            center_y, center_x = h // 2, w // 2
            mask = np.ones((h, w), np.float32)

            for peak_y, peak_x in filtered_peaks:
                cv2.circle(mask, (peak_x, peak_y), adaptive_notch, (0.0,), -1)
                sym_x = center_x - (peak_x - center_x)
                sym_y = center_y - (peak_y - center_y)
                sym_y = int(sym_y)
                sym_x = int(sym_x)
                cv2.circle(mask, ((sym_x), (sym_y)), adaptive_notch, (0.0,), -1)

            f_filtered = f_shifted * mask[:,:,np.newaxis] 
            f_ishifted = np.fft.ifftshift(f_filtered)
            moire_complex = cv2.idft(np.asarray(f_ishifted), flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
            moire_img = np.real(moire_complex)
            moire_img = np.clip(moire_img, 0, 255).astype(np.uint8)
            if spectrum_var > 1000:
                logger.debug("Aplicando filtro bilateral por alta varianza.")
                moire_img = cv2.bilateralFilter(moire_img, d=5, sigmaColor=50, sigmaSpace=50)

            processed_img = np.asarray(moire_img, dtype=np.uint8)
            logger.info("Corrección de moiré aplicada exitosamente.")
            return processed_img
        else:
            logger.debug("No se detectó moiré significativo. No se aplica corrección.")
            processed_img = cropped_img
        
        return processed_img