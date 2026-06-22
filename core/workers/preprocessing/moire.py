# PerfectOCR/core/workflow/preprocessing/moire.py
import cv2
import numpy as np
import logging
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from typing import Dict, Any, Tuple
from core.factory.abstract_worker import PreprocessingAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.utils.image_utils import use_bilateral_filter
from services.output_service import save_croped_image

logger = logging.getLogger(__name__)

class MoireDenoiser(PreprocessingAbstractWorker):
    """Detecta y corrige patrones de moiré."""
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        worker_config = config.get('moire', {})
        self.color = config.get('ink_enhancement', [])["black"]
        self.notch_radius: int = worker_config.get('notch_radius', {})
        self.min_dist_conf: int = worker_config.get('min_distance_from_center', {})
        self.percentile_threshold: int = worker_config.get("percentile_threshold", {})
        self.mean_factor: int = worker_config.get('mean_factor', {})
        self.abs_threshold: int = worker_config.get('abs_threshold', {})
        self.output = config.get("moire_poly", False)

    def preprocess(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """Analiza y corrige el moiré."""
        try:
            logger.debug("Moire empezado conéxito")
            img_obj = manager.get_full_img()
            full_img = img_obj.full_img if img_obj is not None else None
            if full_img is None:
                logger.error(f"No Hay full_img en el Formatter")
                return False
            logger.debug("Full_img obtenida con éxito")

            # full_img = normalice_image(full_img)

            # if full_img is None:
            #     return False

            full_image, corrected = self._detect_moire_patterns(full_img)

            logger.info(f"Valor de corrected: {corrected}")

            if self.output and corrected:
                image_name = manager.workflow.metadata.image_name if manager.workflow else ""
                worker_name = context.get("worker_name") or "moire"
                img_id = f"full_img_{image_name}_{worker_name}"
                save_croped_image(image_name, img_id, full_image, worker_name)

        except Exception as e:
            logger.error(f"Error en moiré: {e}", exc_info=True)
        return True
            
            # Obtener polígonos de las dataclasses
    def _detect_moire_patterns(self, full_gray: np.ndarray[Any, np.dtype[np.uint8]]) -> Tuple[np.ndarray[Any, np.dtype[np.uint8]], bool]:
        """
        Detecta y corrige patrones de moiré en una imagen.
        """
        # 1) Asegurar escala de grises (2D)
        # if full_img.ndim == 3:
        #     full_gray = cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY)
        # else:
        #     full_gray = full_img

        height, width = full_gray.shape  # (y, x)
        max_dim = max(height, width)

        spectrum_var = np.var(full_gray)
        adaptive_notch = max(2, min(10, int(self.notch_radius * (width / 1500))))
        adaptive_min_dist = max(50, min(int(self.min_dist_conf), int((self.min_dist_conf) * (width / 2000))))

        # FFT (2D)
        f_transform = fft2(full_gray)
        f_shifted = fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shifted) + 1)

        # Excluir centro (DC)
        temp_spectrum = magnitude_spectrum.copy().astype(np.float32)
        center_x, center_y = width // 2, height // 2
        cv2.circle(temp_spectrum, (center_x, center_y), adaptive_min_dist, self.color, -1)

        valid = temp_spectrum[temp_spectrum > 0]
        if valid.size == 0:
            return full_gray, False

        mean_energy = np.mean(valid)
        std_energy = np.std(valid)
        skewness = np.mean((valid - mean_energy) ** 3) / (std_energy ** 3) if std_energy > 1e-9 else 0.0

        # Umbral adaptativo (usa config si existe)
        mf = float(self.mean_factor or 3.0)
        if std_energy / max(mean_energy, 1e-9) > 0.5 and skewness > 1.0:
            adaptive_threshold = np.percentile(valid, int(self.percentile_threshold))
            method = "Percentil"
        elif std_energy / max(mean_energy, 1e-9) > 0.3 and skewness < 0.5:
            adaptive_threshold = mean_energy * mf
            method = "Factor"
        else:
            adaptive_threshold = mean_energy + (2.0 * std_energy)
            method = "Absoluto"

        # 2) Picos como MÁXIMOS LOCALES (no todos los > umbral)
        above = (temp_spectrum > adaptive_threshold)
        k = np.ones((3, 3), np.uint8)
        dil = cv2.dilate(temp_spectrum, k)
        local_max = above & (temp_spectrum > dil)

        peaks_yx = np.argwhere(local_max)  # (y, x)

        # Filtrar por distancia al centro
        if peaks_yx.size == 0:
            logger.info("Sin picos detectados (máximos locales).")
            return full_gray, False

        dy = peaks_yx[:, 0] - center_y
        dx = peaks_yx[:, 1] - center_x
        dist_ok = (dx * dx + dy * dy) > (adaptive_min_dist * adaptive_min_dist)
        peaks_yx = peaks_yx[dist_ok]

        # Limitar cantidad (si no, notch masivo y/o resultado inestable)
        max_peaks = 2000
        if peaks_yx.shape[0] > max_peaks:
            vals = temp_spectrum[peaks_yx[:, 0], peaks_yx[:, 1]]
            idx = np.argpartition(vals, -max_peaks)[-max_peaks:]
            peaks_yx = peaks_yx[idx]

        logger.info(
            f"Método: {method}, Umbral: {adaptive_threshold:.3f}, Media: {mean_energy:.3f}, Std: {std_energy:.3f}, "
            f"Picos (local max): {peaks_yx.shape[0]}"
        )

        if peaks_yx.shape[0] == 0:
            return full_gray, False

        # 3) Máscara FFT: 1 = pasa, 0 = notch (NO usar self.color)
        mask = np.ones((height, width), np.float32)

        for (py, px) in peaks_yx:
            px = int(px); py = int(py)
            cv2.circle(mask, (px, py), int(adaptive_notch), self.color, -1)

            sym_x = int(2 * center_x - px)
            sym_y = int(2 * center_y - py)
            if 0 <= sym_x < width and 0 <= sym_y < height:
                cv2.circle(mask, (sym_x, sym_y), int(adaptive_notch), self.color, -1)

        zero_ratio = float(np.mean(mask < 0.5))
        logger.info(f"FFT mask zero ratio: {zero_ratio:.4f} (notches aplicados)")

        f_filtered = f_shifted * mask
        moire_img = np.real(ifft2(ifftshift(f_filtered)))
        moire_img = np.clip(moire_img, 0, 255).astype(np.uint8)

        if spectrum_var > 1000:
            moire_img = use_bilateral_filter(moire_img, d=5, sigma_color=50, sigma_space=50)

        return moire_img, True