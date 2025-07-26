# PerfectOCR/core/workflow/preprocessing/
import cv2
import numpy as np
import logging
from typing import Dict, Any, List
from scipy.fft import fft2, fftshift, ifftshift, ifft2

logger = logging.getLogger(__name__)

class MoireDenoiser:
    """Detecta inclinación con Hough."""
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.corrections = config
        self.denoise_corrections = config.get('denoise', {})

    def _detect_moire_patterns(self, cropped_line: np.ndarray) -> np.ndarray:

        try:
            from mkl import set_num_threads
            set_num_threads(4)
        except ImportError:
            logger.warning("MKL no está disponible, no se puede establecer el número de hilos.")

        moire_corrections = self.denoise_corrections.get('moire', {})
        mode = moire_corrections.get('mode', {})
        percentile_corrections = mode.get('percentile', {})
        factor_corrections = mode.get('factor', {})
        abs_corrections = mode.get('absolute', {})

        notch_radius = int(percentile_corrections.get('notch_radius', 2))
        min_dist = int(percentile_corrections.get('min_distance_from_center', 200))
        mean_factor = factor_corrections.get('mean_factor_threshold', 4)
        abs_threshold = abs_corrections.get('absolute_threshold', 200000000)

        # Ajustes adaptativos
        img_dims = cropped_line.shape
        h, w = img_dims
        max_dim = max(h, w)
        spectrum_var = np.var(cropped_line)
        adaptive_notch = max(2, min(6, int(notch_radius * (spectrum_var / 1000.0) * (max_dim / 1000.0))))
        adaptive_min_dist = max(50, min(300, int(min_dist * (max_dim / 2000.0))))

        # Transformada Fourier
        f_transform = fft2(cropped_line)
        f_shifted = fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shifted) + 1)
                
        # Excluir centro
        temp_spectrum = magnitude_spectrum.copy()
        cv2.circle(temp_spectrum, (h // 2, w // 2), adaptive_min_dist, 0, -1)

        # Estadísticas del espectro para inferencia automática
        valid_spectrum = temp_spectrum[temp_spectrum > 0]
        mean_energy = np.mean(valid_spectrum)
        std_energy = np.std(valid_spectrum)
        skewness = np.mean((valid_spectrum - mean_energy) ** 3) / (std_energy ** 3) if std_energy > 0 else 0

        # Inferencia automática del umbral
        if std_energy / mean_energy > 0.5 and skewness > 1.0:  # Alta variabilidad y asimetría
            adaptive_threshold = np.percentile(valid_spectrum, 98)
            adaptive_threshold = max(1000, min(5000000, adaptive_threshold * (spectrum_var / 100.0)))
            method = "Percentil"
        elif std_energy / mean_energy > 0.3 and skewness < 0.5:  # Variabilidad moderada, distribución uniforme
            adaptive_threshold = mean_energy * 4.0
            adaptive_threshold = max(1000, min(5000000, adaptive_threshold * (spectrum_var / 50.0)))
            method = "Factor"
        else:  # Baja variabilidad o picos fuertes
            adaptive_threshold = 2000000.0
            adaptive_threshold = max(1000, min(5000000, adaptive_threshold * (spectrum_var / 100.0)))
            method = "Absoluto"



        # Detectar picos
        peaks_coords = np.argwhere(magnitude_spectrum > adaptive_threshold)
        filtered_peaks = [(y, x) for y, x in peaks_coords if np.sqrt((y - h//2)**2 + (x - w//2)**2) > adaptive_min_dist]

        # Corrección solo si hay moiré
        if len(filtered_peaks) > 0:
            center_y, center_x = h // 2, w // 2
            mask = np.ones((h, w), np.float32)

            for peak_y, peak_x in filtered_peaks:
                cv2.circle(mask, (peak_x, peak_y), adaptive_notch, 0, -1)
                dist_to_center = np.sqrt((peak_y - center_y)**2 + (peak_x - center_x)**2)
                sym_x = center_x - (peak_x - center_x) * (dist_to_center / max(dist_to_center, 1.0))
                sym_y = center_y - (peak_y - center_y) * (dist_to_center / max(dist_to_center, 1.0))
                cv2.circle(mask, (int(sym_x), int(sym_y)), adaptive_notch, 0, -1)

            f_filtered = f_shifted * mask
            f_ishifted = ifftshift(f_filtered)
            moire_img = np.real(ifft2(f_ishifted))

            # Corrección final con filtro adaptativo
            moire_img = np.clip(moire_img, 0, 255).astype(np.uint8)
            if spectrum_var > 1000:
                moire_img = cv2.bilateralFilter(moire_img, d=5, sigmaColor=50, sigmaSpace=50)
            return moire_img
        else:
            moire_img = cropped_line

        return moire_img