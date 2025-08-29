# PerfectOCR/core/workers/preprocessing/sharp.py
import cv2
import time
import numpy as np
import logging
from typing import Dict, Any, List
from skimage.filters import unsharp_mask
from core.factory.abstract_worker import PreprocessingAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons

logger = logging.getLogger(__name__)

class SharpeningEnhancer(PreprocessingAbstractWorker):

    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = self.config.get('sharpening', {})
        self.enabled_outputs = self.config.get("enabled_outputs", {})
        self.output = self.enabled_outputs.get("sharp_poly", False)

    def preprocess(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """
        Analiza la nitidez de todos los polígonos, decide la corrección con unsharp_mask
        de forma vectorizada y la aplica in-place.
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
                
                analysis = self._analyze_image_for_sharpness(cropped_img_np)
                if analysis:
                    analysis_results.append(analysis)
                    poly_ids_order.append(poly_id)

            if not analysis_results:
                return True

            # 2. Fase de Decisión Vectorizada
            sharpness_metrics = np.array([res['sharpness'] for res in analysis_results], dtype=np.float32)
            variances = np.array([res['variance'] for res in analysis_results], dtype=np.float32)

            adaptive_thresholds = np.maximum(30.0, variances * 0.5)
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

                # logger.debug(f"Poly '{poly_id}': Aplicando Sharpen (Radius: {radius:.2f}, Amount: {amount:.2f})")

                corrected_img = self._apply_sharpening_correction(cropped_img_np, radius, amount)
                polygon.cropped_img.cropped_img = corrected_img
                
                if self.output:
                    self._save_debug_image(context, poly_id, corrected_img)

            total_time = time.time() - start_time
            logger.info(f"Procesamiento Sharpening completado para {len(poly_ids_order)} polígonos en: {total_time:.3f}s")
            return True
        except Exception as e:
            logger.error(f"Error en el procesamiento por lotes de SharpeningEnhancer: {e}", exc_info=True)
            return False

    def _analyze_image_for_sharpness(self, cropped_img_np: np.ndarray[Any, Any]) -> Dict[str, Any]:
        """Calcula métricas de nitidez para una imagen."""
        try:
            sobel = cv2.Sobel(cropped_img_np, cv2.CV_64F, 1, 1, ksize=3)
            sharpness = np.mean(np.abs(sobel))
            variance = np.var(cropped_img_np)
            return {
                "cropped_img_np": cropped_img_np,
                "sharpness": sharpness,
                "variance": variance
            }
        except cv2.error as e:
            logger.warning(f"OpenCV Sobel falló durante el análisis de nitidez: {e}. Se omite la imagen.")
            return {}

    def _apply_sharpening_correction(self, cropped_img_np: np.ndarray[Any, Any], radius: float, amount: float) -> np.ndarray[Any, Any]:
        """Aplica el filtro unsharp_mask a una imagen."""
        sharpened_float = unsharp_mask(cropped_img_np, radius=radius, amount=amount)
        # unsharp_mask devuelve un float en [0, 1], se debe convertir de vuelta a uint8 [0, 255]
        return (np.clip(sharpened_float, 0, 1) * 255).astype(np.uint8)

    def _save_debug_image(self, context: Dict[str, Any], poly_id: str, image: np.ndarray[Any, Any]):
        """Guarda una imagen de depuración si la salida está habilitada."""
        from services.output_service import save_image
        import os
        
        output_paths = context.get("output_paths", [])
        for path in output_paths:
            output_dir = os.path.join(path, "sharp")
            file_name = f"{poly_id}_sharp_debug.png"
            save_image(image, output_dir, file_name)
        
        if output_paths:
            logger.debug(f"Imagen de debug de Sharp para '{poly_id}' guardada")