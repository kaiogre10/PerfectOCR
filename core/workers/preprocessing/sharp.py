# PerfectOCR/core/workers/preprocessing/sharp.py
import numpy as np
import logging
from typing import Dict, Any, List
from skimage.filters import unsharp_mask # type: ignore
from core.factory.abstract_worker import PreprocessingAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons
from core.utils.image_utils import use_sobel
from services.output_service import save_croped_image

logger = logging.getLogger(__name__)

class SharpeningEnhancer(PreprocessingAbstractWorker):

    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        worker_config = config.get('sharpening', {})
        self.sharpness_threshold = worker_config.get("sharpness_threshold")
        self.kernel = worker_config.get("kernel")
        self.output = config.get("sharp_poly", False)

    def preprocess(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """
        Analiza la nitidez de todos los polígonos, decide la corrección con unsharp_mask
        de forma vectorizada y la aplica in-place.
        """
        try:
            if not manager.validate_cropped_img():
                logger.info(f"Sin cropped_img en el formatter")
                return False

            polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
            if not polygons:
                return False

            # 1. Fase de Análisis
            analysis_results: List[Dict[str, Any]] = []
            poly_ids_order: List[str] = []

            for poly_id, polygon in polygons.items():
                cropped_img = polygon.cropped_img.cropped_img if polygon.cropped_img else None
                if cropped_img is None:
                    logger.warning(f"Imagen no encontrada para el polígono '{poly_id}'")
                    continue
                
                cropped_img_np = np.array(cropped_img, dtype=np.uint8)
                if cropped_img_np.size == 0:
                    continue
                
                analysis = self._analyze_image_for_sharpness(cropped_img_np)
                if analysis:
                    analysis_results.append(analysis)
                    poly_ids_order.append(poly_id)

            if not analysis_results:
                return False

            # 2. Fase de Decisión Vectorizada
            sharpness_metrics = np.array([res['sharpness'] for res in analysis_results], dtype=np.float32)
            variances = np.array([res['variance'] for res in analysis_results], dtype=np.float32)

            adaptive_thresholds = np.maximum(self.sharpness_threshold, variances * 0.5)
            needs_correction = sharpness_metrics < adaptive_thresholds

            radii = np.clip(variances - 0.02, 0.5, 2.0)
            amounts = np.clip(variances - 0.03, 1.0, 2.0)

            # 3. Fase de Aplicación
            for idx, poly_id in enumerate(poly_ids_order):
                if not needs_correction[idx]:
                    continue

                polygon = polygons[poly_id]
                cropped_img_np = analysis_results[idx]['cropped_img_np']
                radius = radii[idx]
                amount = amounts[idx]

                corrected_img = self._apply_sharpening_correction(cropped_img_np, radius, amount)
                polygon.cropped_img.cropped_img = corrected_img
                
                if self.output:
                    worker_name = context.get("worker_name") or "sharp"
                    image_name = manager.workflow.metadata.image_name if manager.workflow else ""
                    output_paths = context["output_paths"]
                    save_croped_image(image_name, poly_id, corrected_img, output_paths, worker_name)

            return True
        except Exception as e:
            logger.error(f"Error en el procesamiento por lotes de SharpeningEnhancer: {e}", exc_info=True)
            return False

    def _analyze_image_for_sharpness(self, cropped_img_np: np.ndarray[Any, Any]) -> Dict[str, Any]:
        """Calcula métricas de nitidez para una imagen."""
        try:
            # sobel: np.ndarray[Any, np.dtype[np.float64]] = cv2.Sobel(cropped_img_np, cv2.CV_64F, 1, 1, ksize=self.kernel).astype(dtype=np.float64)
            sharpness = use_sobel(cropped_img_np, self.kernel)
            variance = np.var(cropped_img_np)
            return {
                "cropped_img_np": cropped_img_np,
                "sharpness": sharpness,
                "variance": variance
            }
        except Exception as e:
            logger.warning(f"OpenCV Sobel falló durante el análisis de nitidez: {e}. Se omite la imagen.")
            return {}

    def _apply_sharpening_correction(self, cropped_img_np: np.ndarray[Any, np.dtype[np.uint8]], radius: float, amount: float) -> np.ndarray[Any, np.dtype[np.uint8]]:
        """Aplica el filtro unsharp_mask a una imagen."""
        sharpened_float: np.ndarray[Any, np.dtype[np.float32]] = unsharp_mask(cropped_img_np, radius=radius, amount=amount)
        # unsharp_mask devuelve un float en [0, 1], se debe convertir de vuelta a uint8 [0, 255]
        corrected_img = (np.clip(sharpened_float, 0, 1) * 255).astype(np.uint8)
        return corrected_img
