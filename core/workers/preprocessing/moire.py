# PerfectOCR/core/workflow/preprocessing/
import cv2
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class MoireDenoiser:
    """Detecta y corrige patrones de moiré."""
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.config = config
        
    def _detect_moire_patterns(self, processing_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detecta y corrige patrones de moiré en cada polígono del diccionario,
        modificando 'cropped_img' in-situ.
        Args:
            processing_dict: Diccionario principal con los polígonos
        Returns:
            El mismo diccionario con los 'cropped_img' corregidos si aplica
        """
        polygons = processing_dict.get("polygons", {})
        for poly_data in polygons.values():
            current_image = poly_data.get("cropped_img")
            if current_image is not None:
                # Procesa la imagen y la sobrescribe en el mismo lugar
                poly_data["cropped_img"] = self._detect_moire_single(current_image)
        
        return processing_dict
        
    def _detect_moire_single(self, cropped_img: np.ndarray) -> np.ndarray:

        moire_corrections = self.config
        mode = moire_corrections.get('mode', {})
        percentile_corrections = mode.get('percentile', {})
        factor_corrections = mode.get('factor', {})
        abs_corrections = mode.get('absolute', {})

        notch_radius = int(percentile_corrections.get('notch_radius', 2))
        min_dist = int(percentile_corrections.get('min_distance_from_center', 200))
        mean_factor = factor_corrections.get('mean_factor_threshold', 4)
        abs_threshold = abs_corrections.get('absolute_threshold', 200000000)

        img_dims = cropped_img.shape
        h, w = img_dims
        max_dim = max(h, w)
        spectrum_var = np.var(cropped_img)
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

        if std_energy / mean_energy > 0.5 and skewness > 1.0:  # Alta variabilidad y asimetría
            adaptive_threshold = np.percentile(valid_spectrum, 98)
            adaptive_threshold = max(1000, min(5000000, adaptive_threshold * (spectrum_var / 100.0)))
            mode = "percentile"
        elif std_energy / mean_energy > 0.3 and skewness < 0.5:  # Variabilidad moderada, distribución uniforme
            adaptive_threshold = mean_energy * 4.0
            adaptive_threshold = max(1000, min(5000000, adaptive_threshold * (spectrum_var / 50.0)))
            mode = "factor"
        else:  # Baja variabilidad o picos fuertes
            adaptive_threshold = 2000000.0
            adaptive_threshold = max(1000, min(5000000, adaptive_threshold * (spectrum_var / 100.0)))
            mode = "absolute"

        peaks_coords = np.argwhere(magnitude_spectrum > adaptive_threshold)
        filtered_peaks = [(y, x) for y, x in peaks_coords if np.sqrt((y - h//2)**2 + (x - w//2)**2) > adaptive_min_dist]

        if len(filtered_peaks) > 0:
            center_y, center_x = h // 2, w // 2
            mask = np.ones((h, w), np.float32)

            for peak_y, peak_x in filtered_peaks:
                cv2.circle(mask, (peak_x, peak_y), adaptive_notch, (0.0,), -1)
                sym_x = center_x - (peak_x - center_x)
                sym_y = center_y - (peak_y - center_y)
                cv2.circle(mask, (int(sym_x), int(sym_y)), adaptive_notch, (0.0,), -1)

            f_filtered = f_shifted * mask[:,:,np.newaxis] 
            f_ishifted = np.fft.ifftshift(f_filtered)
            moire_complex = cv2.idft(np.asarray(f_ishifted), flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
            moire_img = np.real(moire_complex)
            moire_img = np.clip(moire_img, 0, 255).astype(np.uint8)
            if spectrum_var > 1000:
                moire_img = cv2.bilateralFilter(moire_img, d=5, sigmaColor=50, sigmaSpace=50)

            moire_poly = moire_img
            return moire_poly
        
        else:
            moire_poly = cropped_img
        
        return moire_poly