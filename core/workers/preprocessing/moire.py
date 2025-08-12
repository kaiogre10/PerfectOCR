# PerfectOCR/core/workflow/preprocessing/moire.py
import cv2
import numpy as np
import logging
import time
from typing import Dict, Any, List
from core.factory.abstract_worker import PreprossesingAbstractWorker
from core.domain.data_models import CroppedImage
from core.domain.data_formatter import DataFormatter

logger = logging.getLogger(__name__)

class MoireDenoiser(PreprossesingAbstractWorker):
    """Detecta y corrige patrones de moiré."""
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.config = config
        
    def preprocess(self, cropped_img: CroppedImage, manager: DataFormatter) -> CroppedImage:
        """
        Detecta y corrige patrones de moiré en cada polígono del diccionario,
        modificando 'cropped_img' in-situ.
        """
        start_time = time.time()
        polygons = manager.get_polygons()
        poly_id = cropped_img.polygon_id  # Obtener el ID del polígono actual
        cropped_geometry = polygons.get(poly_id, {}).get('cropped_geometry', {})
        bbox = cropped_geometry.get("padding_bbox", [])
        if not bbox or len(bbox) < 4:
            logger.error(f"bbox inválido para polígono {poly_id}: {bbox}")
            return cropped_img
        
        if not isinstance(cropped_img.cropped_img, np.ndarray): # type: ignore
            cropped_img.cropped_img = np.array(cropped_img.cropped_img)

        processed_img = self._detect_moire_single(cropped_img.cropped_img, bbox)
            
        cropped_img.cropped_img[...] = processed_img
        
        total_time = time.time() - start_time
        logger.debug(f"Moire completado en: {total_time:.3f}s")
        
        return cropped_img
        
    def _detect_moire_single(self, cropped_img: np.ndarray[Any, Any], bbox: List[float]) -> np.ndarray[Any, Any]:
        moire_corrections = self.config
        mode = moire_corrections.get('mode', {})
        percentile_corrections = mode.get('percentile', {})
        notch_radius = int(percentile_corrections.get('notch_radius', 2))
        min_dist = int(percentile_corrections.get('min_distance_from_center', 200))
                
        w = int(bbox[2] - bbox[0])
        h = int(bbox[3] - bbox[1])
                
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