# PerfectOCR/core/workflow/preprocessing/sp.py
import cv2
import numpy as np
import logging
import time
from typing import Dict, Any, List
from core.factory.abstract_worker import PreprocessingAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons

logger = logging.getLogger(__name__)
    
class DoctorSaltPepper(PreprocessingAbstractWorker):    
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = self.config.get('median_filter', {})
        self.enabled_outputs = self.config.get("enabled_outputs", {})
        self.output = self.enabled_outputs.get("sp_poly", False)
    
    def preprocess(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """
        Analyzes all polygons in a batch, determines the required S&P correction via vectorized operations,
        and applies the correction in-place.
        """
        try:
            start_time = time.time()
            polygons: Dict[str, Polygons] = context.get("polygons", {})
            if not polygons:
                return True

            # 1. Analysis Phase
            metrics: List[Dict[str, Any]] = []
            poly_ids_order: List[str] = []
            for poly_id, polygon in polygons.items():
                # Acceso correcto a la imagen desde la dataclass
                cropped_img = polygon.cropped_img.cropped_img if polygon.cropped_img else None
                if cropped_img is None:
                    logger.warning(f"Imagen no encontrada para el polígono '{poly_id}'")
                    continue
                
                if cropped_img.size == 0:
                    logger.warning(f"Imagen vacía o corrupta en '{poly_id}'")
                    continue
                                
                analysis = self._analyze_image_for_sp(cropped_img)
                if analysis:
                    metrics.append(analysis)
                    poly_ids_order.append(poly_id)

            if not metrics:
                return True

            # 2. Vectorized Decision Phase
            areas = np.array([m['area'] for m in metrics], dtype=np.int32)
            sp_ratios = np.array([m['sp_ratio'] for m in metrics], dtype=np.float32)
            isolated_counts = np.array([m['isolated_count'] for m in metrics], dtype=np.int32)
            min_dims = np.array([min(m['h'], m['w']) for m in metrics], dtype=np.int32)

            cond_small = areas < 2000
            cond_medium = (areas >= 2000) & (areas < 10000)
            
            ratio_thrs = np.select([cond_small, cond_medium], [0.06, 0.03], default=0.015)
            min_isos = np.select([cond_small, cond_medium], [10, 20], default=30)
            ksizes = np.select([cond_small, cond_medium], [3, 3], default=5)
            
            ksizes = np.where(min_dims >= 50, np.maximum(ksizes, 5), ksizes)
            ksizes = np.where(ksizes % 2 == 0, ksizes + 1, ksizes)

            needs_correction = (sp_ratios > ratio_thrs) & (isolated_counts >= min_isos)

            # 3. Application Phase
            for idx, poly_id in enumerate(poly_ids_order):
                if not needs_correction[idx]:
                    continue

                polygon = polygons[poly_id]
                analysis_results = metrics[idx]
                ksize = int(ksizes[idx])

                logger.debug(f"Poly '{poly_id}': Aplicando filtro S&P con k-size: {ksize}")

                corrected_img = self._apply_sp_correction(
                    analysis_results,
                    ksize
                )
                
                polygon.cropped_img.cropped_img = corrected_img
                
                if self.output:
                    self._save_debug_image(context, poly_id, corrected_img)

            total_time = time.time() - start_time
            logger.info(f"S&P batch completado para {len(poly_ids_order)} polígonos en: {total_time:.3f}s")
            return True
        except Exception as e:
            logger.error(f"Error en el procesamiento por lotes de S&P: {e}", exc_info=True)
            return False

    def _analyze_image_for_sp(self, cropped_img: np.ndarray[Any, np.dtype[np.uint8]]) -> Dict[str, Any]:
        h, w = cropped_img.shape[:2]
        area = h * w
        if area == 0:
            return {}

        p1, p99 = np.percentile(cropped_img, [1, 99])
        low, high = int(max(0, p1)), int(min(255, p99))
        
        extreme_mask = (cropped_img <= low) | (cropped_img >= high)
        sp_ratio = np.count_nonzero(extreme_mask) / area

        kernel = np.ones((3, 3), np.uint8)
        neighbor_count = cv2.filter2D(extreme_mask.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_REPLICATE)
        isolated_mask = extreme_mask & (neighbor_count <= 1)
        isolated_count = np.count_nonzero(isolated_mask)

        return {
            "original_img": cropped_img,
            "h": h, "w": w, "area": area,
            "sp_ratio": sp_ratio,
            "isolated_count": isolated_count,
            "extreme_mask": extreme_mask,
            "sobel_before": np.mean(np.abs(cv2.Sobel(cropped_img, cv2.CV_64F, 1, 1, ksize=3)))
        }

    def _apply_sp_correction(self, analysis: Dict[str, Any], ksize: int) -> np.ndarray[Any, np.dtype[np.uint8]]:
        original_img = analysis['original_img']
        filtered = cv2.medianBlur(original_img, ksize)
        
        result = original_img.copy()
        result[analysis['extreme_mask']] = filtered[analysis['extreme_mask']]

        sobel_after = np.mean(np.abs(cv2.Sobel(result, cv2.CV_64F, 1, 1, ksize=3)))
        
        if sobel_after < 0.85 * analysis['sobel_before']:
            logger.debug("Corrección S&P revertida por pérdida de nitidez.")
            return original_img
        
        return result

    def _save_debug_image(self, context: Dict[str, Any], poly_id: str, image: np.ndarray[Any, np.dtype[np.uint8]]):
        from services.output_service import save_image
        import os
        
        output_paths = context.get("output_paths", [])
        for path in output_paths:
            output_dir = os.path.join(path, "sp")
            file_name = f"{poly_id}_sp_debug.png"
            save_image(image, output_dir, file_name)
        
        if output_paths:
            logger.debug(f"Imagen de debug de S&P para '{poly_id}' guardada en {len(output_paths)} ubicaciones.")