# PerfectOCR/core/workers/preprocessing/gauss.py
import cv2
import numpy as np
import logging
import time
from typing import Dict, Any, List
from core.factory.abstract_worker import PreprocessingAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons

logger = logging.getLogger(__name__)

class GaussianDenoiser(PreprocessingAbstractWorker):

    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = self.config.get('bilateral_params', {})
        self.enabled_outputs = self.config.get("enabled_outputs", {})
        self.output = self.enabled_outputs.get("gauss_poly", False)
        
        # Parámetros del filtro leídos desde la configuración una sola vez
        self.d = self.worker_config.get('d', 9)
        self.sigma_color = self.worker_config.get('sigma_color', 75)
        self.sigma_space = self.worker_config.get('sigma_space', 75)
        self.gauss_threshold = self.worker_config.get('laplacian_variance_threshold', 100)

    def preprocess(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """
        Analiza todos los polígonos en un lote para detectar ruido Gaussiano, determina la corrección
        necesaria mediante operaciones vectorizadas y la aplica in-place.
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
                    logger.warning(f"Imagen vacía para el polígono '{poly_id}'")
                    continue
                
                analysis = self._analyze_image_for_gauss(cropped_img_np)
                if analysis:
                    analysis_results.append(analysis)
                    poly_ids_order.append(poly_id)

            if not analysis_results:
                return True

            # 2. Fase de Decisión Vectorizada
            laplacian_variances = np.array([res['laplacian_var'] for res in analysis_results], dtype=np.float32)
            needs_correction = laplacian_variances < self.gauss_threshold

            # 3. Fase de Aplicación
            for idx, poly_id in enumerate(poly_ids_order):
                if not needs_correction[idx]:
                    continue

                polygon = polygons[poly_id]
                original_img = analysis_results[idx]['original_img']

                logger.debug(f"Poly '{poly_id}': Aplicando filtro bilateral (Varianza: {laplacian_variances[idx]:.2f})")

                corrected_img = self._apply_gauss_correction(original_img)
                
                polygon.cropped_img.cropped_img = corrected_img
                
                if self.output:
                    self._save_debug_image(context, poly_id, corrected_img)

            total_time = time.time() - start_time
            logger.info(f"Procesamiento Gaussiano completado para {len(poly_ids_order)} polígonos en: {total_time:.3f}s")
            return True
        except Exception as e:
            logger.error(f"Error en el procesamiento por lotes de GaussianDenoiser: {e}", exc_info=True)
            return False

    def _analyze_image_for_gauss(self, cropped_img: np.ndarray[Any, Any]) -> Dict[str, Any]:
        """Calcula la varianza del Laplaciano para una imagen."""
        try:
            # La varianza del Laplaciano es sensible a la profundidad de bits, CV_64F es estándar.
            laplacian_var = cv2.Laplacian(cropped_img, cv2.CV_64F).var()
            return {
                "original_img": cropped_img,
                "laplacian_var": laplacian_var
            }
        except cv2.error as e:
            logger.warning(f"OpenCV falló durante el análisis Gaussiano: {e}. Se omite la imagen.")
            return {}

    def _apply_gauss_correction(self, original_img: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Aplica el filtro bilateral a una imagen."""
        return cv2.bilateralFilter(original_img, self.d, self.sigma_color, self.sigma_space)

    def _save_debug_image(self, context: Dict[str, Any], poly_id: str, image: np.ndarray[Any, Any]):
        """Guarda una imagen de depuración si la salida está habilitada."""
        from services.output_service import save_image
        import os
        
        output_paths = context.get("output_paths", [])
        for path in output_paths:
            output_dir = os.path.join(path, "gauss")
            file_name = f"{poly_id}_gauss_debug.png"
            save_image(image, output_dir, file_name)
        
        if output_paths:
            logger.debug(f"Imagen de debug Gauss para '{poly_id}' guardada en {len(output_paths)} ubicaciones.")
