# PerfectOCR/core/workers/preprocessing/ink_enhancer.py
import cv2
import numpy as np
import logging
import time
from typing import Dict, Any, List
from core.factory.abstract_worker import ImagePrepAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.utils.image_analizer import extract_cc_metrics

logger = logging.getLogger(__name__)

class InkEnhancer(ImagePrepAbstractWorker):
    """Worker especializado en restaurar texto con tinta gastada o de baja intensidad."""

    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get('ink_enhancement', {})
        self.bin_interval = config["bin_interval"]
        self.kernel_threshold: int = self.worker_config.get("kernel_threshold", 3)
        self.area_threshold: int = self.worker_config.get("area_threshold", 12)
        self.iterations: int = self.worker_config.get("iterations", 2)
        self.faded_threshold = self.worker_config.get('faded_detection_threshold')
        self.output = config.get("ink_poly", False)

    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """Detecta y restaura texto con tinta gastada."""
        try:
            start_time = time.perf_counter()
            logger.info("Mejoramiento de tinta empezado con éxito")
            image_name = manager.workflow.metadata.image_name if manager.workflow else ""

            img_obj = manager.get_full_img()
            full_img = img_obj.full_img if img_obj is not None else None
            if full_img is None:
                logger.error(f"No Hay full_img en el Formatter")
                return False
                    
            _, full_bin_img = extract_cc_metrics(full_img, worker_config={}, binarice=True)
            if full_bin_img is None:
                return False
            
            if self.output:
                from services.output_service import save_croped_image
                worker_name = context.get("worker_name") or "inker"
                output_paths = context["output_paths"]
                img_id = f"full_img_{image_name}_{worker_name}"
                save_croped_image(image_name, img_id, full_bin_img, output_paths, worker_name, method="deskewed")

            full_bin_img_rest = self._restore_faded_ink(full_bin_img)

            _, full_bin_img = extract_cc_metrics(full_bin_img, worker_config={}, binarice=False)
            if full_bin_img is None:
                return False
                    
            logger.debug(f"Restauración de tinta completada para '{image_name}' en: {time.perf_counter() - start_time:.6f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Error en InkEnhancer: {e}", exc_info=True)
            return False

    def _restore_faded_ink(self, full_bin_img: np.ndarray[Any, np.dtype[np.uint8]]) -> np.ndarray[Any, np.dtype[np.uint8]]:
        """Restaura la intensidad del texto con tinta gastada."""
        # 1. Estiramiento adaptativo del histograma
        p1, p99 = np.percentile(img, [1, 99])
        if p99 > p1:
            stretched = np.clip((img - p1) * (255 / (p99 - p1)), 0, 255)
        else:
            stretched = img.astype(np.float32)

        # 2. Gamma correction adaptativa basada en el score de desvanecimiento
        gamma = 0.5 + (1.0 - faded_score) * 0.3  # Gamma entre 0.5-0.8
        gamma_corrected = np.power(stretched / 255.0, gamma) * 255

        # 3. Realce de contraste local usando operador unsharp mask
        gaussian_blurred = cv2.GaussianBlur(gamma_corrected, (3, 3), 1.0)
        unsharp_strength = faded_score * 0.8  # Más fuerza para tinta más gastada
        unsharp_enhanced = gamma_corrected + unsharp_strength * (gamma_corrected - gaussian_blurred)

        # 4. Aplicar CLAHE localizado para mejorar contraste
        clahe = cv2.createCLAHE(clipLimit=2.0 + faded_score * 2.0, tileGridSize=(4, 4))
        final_enhanced = clahe.apply(np.clip(unsharp_enhanced, 0, 255).astype(np.uint8))

        # 5. Post-procesamiento: suavizado ligero para reducir artefactos
        final_enhanced = cv2.bilateralFilter(final_enhanced, 5, 20, 20)

        return final_enhanced
