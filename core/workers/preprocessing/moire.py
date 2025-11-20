# PerfectOCR/core/workflow/preprocessing/moire.py
import cv2
import numpy as np
import logging
import time
from typing import Dict, Any, Tuple, List
from core.factory.abstract_worker import PreprocessingAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons

logger = logging.getLogger(__name__)

class MoireDenoiser(PreprocessingAbstractWorker):
    """Detecta y corrige patrones de moiré."""
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get('moire', {})
        self.notch_radius_conf = self.worker_config.get('notch_radius')
        self.min_dist_conf = self.worker_config.get('min_distance_from_center')
        self.enabled_outputs = config.get("image_load_outputs", {})
        self.output = self.enabled_outputs.get("moire_poly", False)

    def preprocess(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """Analiza y corrige el moiré."""
        try:
            logger.debug("Moire empezado conéxito")
            start_time = time.time()
            
            if not manager.validate_cropped_img():
                logger.info(f"Sin cropped_img en el formatter")
                return False
                
            logger.debug("Polygonos revisados")
            
            # Obtener polígonos de las dataclasses
            polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
            if not polygons:
                return False

            # Fase de Análisis: Recopilar métricas
            metrics: List[Dict[str, Any]] = []
            poly_ids_order: List[str] = []
            
            for poly_id, polygon in polygons.items():
                # Acceso correcto a la imagen desde la dataclass
                cropped_img = polygon.cropped_img.cropped_img if polygon.cropped_img else None
                if cropped_img is None:
                    logger.warning(f"Imagen no encontrada para el polígono '{poly_id}'")
                    continue
                
                h = polygon.cropedd_geometry.croppy_dims.get("poly_height") or cropped_img.shape[0]
                w = polygon.cropedd_geometry.croppy_dims.get("poly_width") or cropped_img.shape[1]
                
                analysis = self._analyze_image_for_moire(cropped_img, (h, w))
                if analysis:
                    metrics.append(analysis)
                    poly_ids_order.append(poly_id)

            if not metrics:
                logger.error("No se encontraron imágenes válidas para el análisis de moiré.")
                return True

            # 3. Fase de Decisión Vectorizada (sin cambios)
            spectrum_vars = np.array([m['spectrum_var'] for m in metrics], dtype=np.float32)
            mean_energies = np.array([m['mean_energy'] for m in metrics], dtype=np.float32)
            std_energies = np.array([m['std_energy'] for m in metrics], dtype=np.float32)
            skewness_values = np.array([m['skewness'] for m in metrics], dtype=np.float32)
            valid_spectrums = [m['valid_spectrum'] for m in metrics]
            safe_mean_energies = np.where(mean_energies == 0, 1.0, mean_energies)
            cond_percentile = (std_energies / safe_mean_energies > 0.5) & (skewness_values > 1.0)
            cond_factor = (std_energies / safe_mean_energies > 0.3) & (skewness_values < 0.5)

            thresholds_percentile = np.array([np.percentile(vs, 98) if vs.size > 0 else 0 for vs in valid_spectrums], dtype=np.float32)
            
            adaptive_thresholds = np.select(
                [cond_percentile, cond_factor],
                [
                    np.maximum(1000, np.minimum(5000000, thresholds_percentile * (spectrum_vars / 100.0))),
                    np.maximum(1000, np.minimum(5000000, mean_energies * 4.0 * (spectrum_vars / 50.0)))
                ],
                default=np.maximum(1000, np.minimum(5000000, 2000000.0 * (spectrum_vars / 100.0)))
            )

            # 4. Fase de Aplicación (modificando las dataclasses"in-line")
            for idx, poly_id in enumerate(poly_ids_order):
                polygon = polygons[poly_id]
                analysis = metrics[idx]
                
                # Obtener la imagen original de la dataclass para la corrección
                original_img_np = polygon.cropped_img.cropped_img 
                
                threshold = adaptive_thresholds[idx]
                # logger.debug(f"Poly '{poly_id}': Modo de corrección '{mode}', Threshold: {threshold:.2f}")

                corrected_img: np.ndarray[Any, np.dtype[np.uint8]] = self._apply_moire_correction(
                    original_img_np,
                    analysis,
                    threshold,
                )
                
                polygon.cropped_img.cropped_img = corrected_img
                
                if self.output:
                    from services.output_service import save_croped_image
                    worker_name = context.get("worker_name") or "moire"
                    image_name = manager.workflow.metadata.image_name if manager.workflow else ""
                    output_paths = context["output_paths"]
                    save_croped_image(image_name, poly_id, corrected_img, output_paths, worker_name, method=worker_name)
                    
            total_time = time.time() - start_time
            logger.debug(f"Moire batch completado para {len(poly_ids_order)} polígonos en: {total_time:.3f}s")
            return True
            
        except Exception as e:
            logger.error(f"Error en el procesamiento Moire: {e}", exc_info=True)
            return False

    def _analyze_image_for_moire(self, cropped_img: np.ndarray[Any, np.dtype[np.uint8]], img_dims: Tuple[int, int]) -> Dict[str, Any]:
        try:
            h, w = img_dims
            img_float: np.ndarray[Any, np.dtype[np.float32]] = np.asanyarray(cropped_img, dtype=np.float32)
            f_transform  = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
            f_shifted = np.fft.fftshift(f_transform)
            
            magnitude_spectrum = cv2.magnitude(f_shifted[:, :, 0], f_shifted[:, :, 1])
            magnitude_spectrum = 20 * np.log(magnitude_spectrum + 1)
            
            min_dist_conf = self.worker_config.get('min_distance_from_center')
            adaptive_min_dist = max(50, min(300, int(min_dist_conf * (max(h, w) / 2000.0))))
            
            temp_spectrum = magnitude_spectrum.copy()
            cv2.circle(temp_spectrum, (w // 2, h // 2), adaptive_min_dist, (0.0,), -1)
            
            valid_spectrum = temp_spectrum[temp_spectrum > 0]
            mean_energy = np.mean(valid_spectrum) if valid_spectrum.size > 0 else 0.0
            std_energy = np.std(valid_spectrum) if valid_spectrum.size > 0 else 0.0
            skewness = np.mean((valid_spectrum - mean_energy) ** 3) / (std_energy ** 3) if std_energy > 0 else 0.0

            return {
                "spectrum_var": np.var(cropped_img),
                "mean_energy": mean_energy,
                "std_energy": std_energy,
                "skewness": skewness,
                "magnitude_spectrum": magnitude_spectrum,
                "f_shifted": f_shifted,
                "img_dims": img_dims,
                "valid_spectrum": valid_spectrum
            }
            
        except Exception as e:
            logger.error(f"Error analizando poligono para moire: {e}", exc_info=True)
            return {}

    def _apply_moire_correction(self, original_img_np: np.ndarray[Any, np.dtype[np.uint8]], analysis: Dict[str, Any], adaptive_threshold: float) -> np.ndarray[Any, np.dtype[np.uint8]]:
        h, w = analysis['img_dims']
        max_dim = max(h, w)
        spectrum_var = analysis['spectrum_var']
        
        adaptive_notch = max(2, min(6, int(self.notch_radius_conf * (spectrum_var / 1000.0) * (max_dim / 1000.0))))
        adaptive_min_dist = max(50, min(300, int(self.min_dist_conf * (max_dim / 2000.0))))

        peaks_coords = np.argwhere(analysis['magnitude_spectrum'] > adaptive_threshold)
        filtered_peaks = [(y, x) for y, x in peaks_coords if np.sqrt((y - h//2)**2 + (x - w//2)**2) > adaptive_min_dist]

        if len(filtered_peaks) > 0:
            mask = np.ones((h, w), np.float32)
            center_y, center_x = h // 2, w // 2

            for peak_y, peak_x in filtered_peaks:
                cv2.circle(mask, (peak_x, peak_y), adaptive_notch, (0.0,), -1)
                sym_x = int(center_x - (peak_x - center_x))
                sym_y = int(center_y - (peak_y - center_y))
                cv2.circle(mask, (sym_x, sym_y), adaptive_notch, (0.0,), -1)

            f_filtered = analysis['f_shifted'] * mask[:, :, np.newaxis]
            f_ishifted = np.fft.ifftshift(f_filtered)
            moire_complex = cv2.idft(f_ishifted, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
            moire_img = np.clip(np.real(moire_complex), 0, 255).astype(np.uint8)

            if spectrum_var > 1000:
                moire_img = cv2.bilateralFilter(moire_img, d=5, sigmaColor=50, sigmaSpace=50)
            
            corrected_img =np.array(moire_img, dtype=np.uint8)
            return corrected_img 
        else:
            corrected_img = np.array(original_img_np, dtype=np.uint8)
            return corrected_img