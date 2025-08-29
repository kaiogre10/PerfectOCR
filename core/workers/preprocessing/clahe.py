# PerfectOCR/core/workers/preprocessing/clahe.py
import cv2
import numpy as np
import logging
import time
from typing import Dict, Any, List, Tuple
from core.factory.abstract_worker import PreprocessingAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons

logger = logging.getLogger(__name__)

class ClaherEnhancer(PreprocessingAbstractWorker):

    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = self.config.get('contrast', {})
        self.enabled_outputs = self.config.get("enabled_outputs", {})
        self.output = self.enabled_outputs.get("clahe_poly", False)
        
        # Parámetros de configuración
        global_clahe_corrects = self.worker_config.get('global', {})
        self.contrast_threshold = global_clahe_corrects.get('contrast_threshold', 50.0)
        self.page_dimensions = global_clahe_corrects.get('dimension_thresholds_px', [1000, 2500])
        self.grid_maps = global_clahe_corrects.get('grid_sizes_map', [[6, 6], [8, 8], [10, 10]])

    def preprocess(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """
        Analiza el contraste de todos los polígonos, decide la corrección con CLAHE de forma
        vectorizada y la aplica in-place.
        """
        try:
            start_time = time.time()
            polygons: Dict[str, Polygons] = context.get("polygons", {})
            if not polygons:
                return True

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
                
                analysis = self._analyze_image_for_contrast(cropped_img_np)
                if analysis:
                    analysis_results.append(analysis)
                    poly_ids_order.append(poly_id)

            if not analysis_results:
                return True

            # 2. Fase de Decisión Vectorizada
            stds = np.array([res['std'] for res in analysis_results], dtype=np.float32)
            variances = np.array([res['var'] for res in analysis_results], dtype=np.float32)
            dyn_ranges = np.array([res['dyn_range'] for res in analysis_results], dtype=np.float32)
            heights = np.array([res['h'] for res in analysis_results], dtype=np.int32)
            widths = np.array([res['w'] for res in analysis_results], dtype=np.int32)

            dynamic_intervals = np.maximum(30.0, variances * 0.6)
            adaptive_thresholds = np.where(dyn_ranges > self.contrast_threshold, dynamic_intervals, 20.0)
            
            needs_correction = stds < adaptive_thresholds

            max_dims = np.maximum(heights, widths)
            cond_small = max_dims < self.page_dimensions[0]
            cond_medium = max_dims < self.page_dimensions[1]
            
            # Vectoriza los tamaños de grid para cada polígono
            grid_small = np.tile(self.grid_maps[0], (len(max_dims), 1))
            grid_medium = np.tile(self.grid_maps[1], (len(max_dims), 1))
            grid_large = np.tile(self.grid_maps[2], (len(max_dims), 1))

            grid_sizes = np.where(cond_small[:, None], grid_small,
                          np.where(cond_medium[:, None], grid_medium, grid_large))
            clip_limits = np.clip(dyn_ranges * 0.01, 1.0, 3.0)

            # 3. Fase de Aplicación
            for idx, poly_id in enumerate(poly_ids_order):
                if not needs_correction[idx]:
                    continue

                polygon = polygons[poly_id]
                original_img = analysis_results[idx]['original_img']
                grid_size = tuple(grid_sizes[idx])
                clip_limit = clip_limits[idx]

                # logger.debug(f"Poly '{poly_id}': Aplicando CLAHE (Grid: {grid_size}, Clip: {clip_limit:.2f})")

                corrected_img = self._apply_clahe_correction(original_img, clip_limit, grid_size)
                polygon.cropped_img.cropped_img = corrected_img
                
                if self.output:
                    self._save_debug_image(context, poly_id, corrected_img)

            total_time = time.time() - start_time
            logger.info(f"Procesamiento CLAHE completado para {len(poly_ids_order)} polígonos en: {total_time:.3f}s")
            return True
        except Exception as e:
            logger.error(f"Error en el procesamiento por lotes de ClaherEnhancer: {e}", exc_info=True)
            return False

    def _analyze_image_for_contrast(self, cropped_img: np.ndarray[Any, np.dtype[np.uint8]]) -> Dict[str, Any]:
        """Calcula métricas de contraste para una imagen."""
        h, w = cropped_img.shape[:2]
        return {
            "original_img": cropped_img,
            "h": h, "w": w,
            "std": np.std(cropped_img),
            "var": np.var(cropped_img),
            "dyn_range": np.max(cropped_img) - np.min(cropped_img)
        }

    def _apply_clahe_correction(self, original_img: np.ndarray[Any, np.dtype[np.uint8]], clip_limit: float, grid_size: Tuple[ Any, ...]) -> np.ndarray[Any, Any]:
        """Aplica el filtro CLAHE a una imagen."""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        return clahe.apply(original_img)

    def _save_debug_image(self, context: Dict[str, Any], poly_id: str, image: np.ndarray[Any, np.dtype[np.uint8]]):
        """Guarda una imagen de depuración si la salida está habilitada."""
        from services.output_service import save_image
        import os
        
        output_paths = context.get("output_paths", [])
        for path in output_paths:
            output_dir = os.path.join(path, "clahe")
            file_name = f"{poly_id}_clahe_debug.png"
            save_image(image, output_dir, file_name)
        
        if output_paths:
            logger.debug(f"Imagen de debug de CLAHE para '{poly_id}' guardada en {len(output_paths)} ubicaciones.")