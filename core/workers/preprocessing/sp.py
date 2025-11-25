# PerfectOCR/core/workflow/preprocessing/sp.py
import cv2
import numpy as np
import logging
import time
from typing import Dict, Any, List
from core.factory.abstract_worker import PreprocessingAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons
from core.utils.image_utils import use_sobel
from core.utils.image_analizer import extract_cc_metrics

logger = logging.getLogger(__name__)
    
class DoctorSaltPepper(PreprocessingAbstractWorker):    
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get('sp_config', {})
        self.sobel_threshold = self.worker_config.get("sobel_threshold")
        self.salt_pepper_low = self.worker_config.get("salt_pepper_low")
        self.salt_pepper_high = self.worker_config.get("salt_pepper_high")
        self.salt_pepper_threshold = self.worker_config.get("salt_pepper_threshold")
        self.kernel_size = self.worker_config.get("kernel_size")
        self.output = config.get("sp_poly", False)
    
    def preprocess(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """
        Analyzes all polygons in a batch, determines the required S&P correction via vectorized operations,
        and applies the correction in-place.
        """
        try:
            start_time = time.time()
            
            if not manager.validate_cropped_img():
                logger.info(f"Sin cropped_img en el formatter")
                return False
                
            logger.debug("Polygonos revisados")
            polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
            if not polygons:
                return False
            
            # 1. Analysis Phase
            metrics: List[Dict[str, Any]] = []
            poly_ids_order: List[str] = []
            corrected_poly = 0
            for poly_id, polygon in polygons.items():
                # Acceso correcto a la imagen desde la dataclass
                cropped_img = polygon.cropped_img.cropped_img if polygon.cropped_img else None
                if cropped_img is None:
                    logger.warning(f"Imagen no encontrada para el polígono '{poly_id}'")
                    continue
               
                corrected2 = self.analized_blobs(cropped_img.copy())
                
                analysis = self._analyze_image_for_sp(cropped_img)
                if analysis:
                    metrics.append(analysis)
                    poly_ids_order.append(poly_id)

            if not metrics:
                return True

            # 2. Vectorized Decision Phase
            areas = np.array([m['area'] for m in metrics], dtype=np.uint32)
            sp_ratios = np.array([m['sp_ratio'] for m in metrics], dtype=np.float32)
            isolated_counts = np.array([m['isolated_count'] for m in metrics], dtype=np.uint32)
            min_dims = np.array([min(m['h'], m['w']) for m in metrics], dtype=np.uint32)

            cond_small = areas < 1500  # Reducido de 2000
            cond_medium = (areas >= 1500) & (areas < 5000)  # Ajustado de 2000-10000
            
            # Ajuste de umbrales y ksizes para tamaños más realistas
            ratio_thrs = np.select([cond_small, cond_medium], [0.05, 0.025], default=self.salt_pepper_threshold)
            min_isos = np.select([cond_small, cond_medium], [8, 15], default=25)
            ksizes = np.select([cond_small, cond_medium], [3, 3], default=5)
            
            ksizes = np.where(min_dims > 40, np.maximum(ksizes, 5), ksizes) # Reducido de 50
            ksizes = np.where(ksizes % 2 == 0, ksizes + 1, ksizes)

            # logger.info(f"DPI: {dpi}")
            # logger.info(f"AREAS: {areas}")

            needs_correction = (sp_ratios > ratio_thrs) & (isolated_counts > min_isos)

            # 3. Application Phase
            for idx, poly_id in enumerate(poly_ids_order):
                if not needs_correction[idx]:
                    continue

                corrected_poly += 1

                polygon = polygons[poly_id]
                analysis_results = metrics[idx]
                ksize = int(ksizes[idx])

                # logger.info(f"{poly_id}:")

                corrected_img = self._apply_sp_correction(
                    analysis_results,
                    ksize
                )
                
                polygon.cropped_img.cropped_img = corrected_img
               
                
                
                if self.output:
                    from services.output_service import save_croped_image
                    worker_name = context.get("worker_name") or "sp"
                    image_name = manager.workflow.metadata.image_name if manager.workflow else ""
                    output_paths = context["output_paths"]
                    save_croped_image(image_name, poly_id, corrected_img, output_paths, worker_name, method=worker_name)

            total_time = time.time() - start_time
            logger.info(f"Corregidos: {corrected_poly}/{len(poly_ids_order)} polígonos en: {total_time:.6f}s")
            return True
        
        except Exception as e:
            logger.error(f"Error en el procesamiento por lotes de S&P: {e}", exc_info=True)
            return False

    def _analyze_image_for_sp(self, cropped_img: np.ndarray[Any, np.dtype[np.uint8]]) -> Dict[str, Any]:
        h, w = cropped_img.shape[:2]
        if h == 0 or w == 0:
            logger.warning("imagen no válida")
            return {}
        
        area = h * w

        p1, p99 = np.percentile(cropped_img, [1, 99])
        low, high = int(max(0, p1)), int(min(self.salt_pepper_high, p99))
        
        extreme_mask = (cropped_img < low).astype(np.uint8) | (cropped_img > high).astype(np.uint8)
        sp_ratio = np.count_nonzero(extreme_mask) / area

        kernel = np.ones((3, 3), np.uint8)
        neighbor_count = cv2.filter2D(extreme_mask.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_REPLICATE)
        isolated_mask = extreme_mask & (neighbor_count < 2)
        isolated_count = np.count_nonzero(isolated_mask)
        sobel_before = use_sobel(cropped_img, self.kernel_size)
        # sobel_before = np.mean(np.abs(cv2.Sobel(cropped_img, cv2.CV_64F, 1, 1, ksize=self.kernel_size)))

        return {
            "original_img": cropped_img,
            "h": h, "w": w, "area": area,
            "sp_ratio": sp_ratio,
            "isolated_count": isolated_count,
            "extreme_mask": extreme_mask,
            "sobel_before": sobel_before
        }

    def _apply_sp_correction(self, analysis: Dict[str, Any], ksize: int) -> np.ndarray[Any, np.dtype[np.uint8]]:
        original_img: np.ndarray[Any, np.dtype[np.uint8]] = analysis['original_img']
        filtered = cv2.medianBlur(original_img, ksize)
        
        result = original_img.copy()
        result[analysis['extreme_mask']] = filtered[analysis['extreme_mask']]
        
        sobel_after = use_sobel(result, self.kernel_size)
        sobel_before = analysis["sobel_before"]
        sobel_thr = self.sobel_threshold * sobel_before
        # logger.info(f"SOBEL BEFORE: {sobel_before} y AFTER: {sobel_after} | condición: {sobel_thr}")

        if sobel_after > sobel_thr:
            # logger.debug("Corregido S&P")
            return result
        
        else:
            logger.info(f"Corrección S&P revertida por pérdida de nitidez")
            return original_img

    def analized_blobs(self, cropped_img: np.ndarray[Any, np.dtype[np.uint8]]) -> np.ndarray[Any, np.dtype[np.uint8]]:
        cc, contours = extract_cc_metrics(cropped_img, worker_config={}, binarice=True)
        logger.info(f"Shaoe: {cc.shape}")
        mask = cc[:, 1] < self.salt_pepper_low
        blobs = cc[mask]
        logger.info(f" blobs {blobs[:, 1]}")
        noise = contours.get("")