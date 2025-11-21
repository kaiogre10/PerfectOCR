# PerfectOCR/core/workers/preprocessing/gauss.py
import cv2
import numpy as np
import logging
from typing import Dict, Any, List
from core.factory.abstract_worker import PreprocessingAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons
from core.utils.image_utils import use_bilateral_filter

logger = logging.getLogger(__name__)

class GaussianDenoiser(PreprocessingAbstractWorker):

    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get('gauss_params', {})
        self.gauss_threshold = self.worker_config.get('laplacian_variance_threshold')
        self.output = config.get("gauss_poly", False)

    def preprocess(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """
        Analiza todos los polígonos en un lote para detectar ruido Gaussiano, determina la corrección
        necesaria mediante operaciones vectorizadas y la aplica in-place.
        """
        try:            
            if not manager.validate_cropped_img():
                logger.error(f"Sin cropped_img en el formatter")
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

                analysis = self._analyze_image_for_gauss(cropped_img)
                if analysis:
                    analysis_results.append(analysis)
                    poly_ids_order.append(poly_id)
                
            if not analysis_results:
                return True

            # 2. Fase de Decisión Vectorizada
            laplacian_variances = np.array([res['laplacian_var'] for res in analysis_results], dtype=np.float32)
            # Se invierte la lógica: un valor alto de varianza del Laplaciano indica bordes nítidos y/o ruido.
            needs_correction = laplacian_variances > self.gauss_threshold

            # 3. Fase de Aplicación
            for idx, poly_id in enumerate(poly_ids_order):
                if not needs_correction[idx]:
                    continue

                polygon = polygons[poly_id]
                original_img = analysis_results[idx]['original_img']

                # logger.info(f"'{poly_id}': Gauss (Varianza: {laplacian_variances[idx]:.4f})")

                corrected_img = self._apply_gauss_correction(original_img)
                
                polygon.cropped_img.cropped_img = corrected_img
                
                if self.output:
                    from services.output_service import save_croped_image
                    worker_name = context.get("worker_name") or "gauss"
                    image_name = manager.workflow.metadata.image_name if manager.workflow else ""
                    output_paths = context["output_paths"]
                    save_croped_image(image_name, poly_id, corrected_img, output_paths, worker_name, method=worker_name)

            logger.debug(f"Procesamiento Gaussiano completado para {len(poly_ids_order)} polígonos")

            return True
        
        except Exception as e:
            logger.error(f"Error en el procesamiento por lotes de GaussianDenoiser: {e}", exc_info=True)
            return False

    def _analyze_image_for_gauss(self, cropped_img_np: np.ndarray[Any, np.dtype[np.uint8]]) -> Dict[str, np.ndarray[Any, np.dtype[np.uint8]]]:
        """Calcula la varianza del Laplaciano para una imagen."""
        try:
            # La varianza del Laplaciano es sensible a la profundidad de bits, CV_64F es estándar.
            laplacian_var = cv2.Laplacian(cropped_img_np, cv2.CV_64F).var()
            return {
                "original_img": cropped_img_np,
                "laplacian_var": laplacian_var
            }
        
        except cv2.error as e:
            logger.warning(f"OpenCV falló durante el análisis Gaussiano: {e}. Se omite la imagen.")
            return {}

    def _apply_gauss_correction(self, original_img: np.ndarray[Any, np.dtype[np.uint8]]) -> np.ndarray[Any, np.dtype[np.uint8]]:
        """Aplica el filtro bilateral a una imagen con parámetros adaptativos."""
        h, w = original_img.shape[:2]

        # 1. Diámetro del vecindario (d) adaptativo al tamaño de la imagen.
        # Se asegura que sea un entero impar y al menos 3.
        d = max(3, int(2 * round(min(h, w) / 150) + 1))

        # 2. Sigma color adaptativo a la desviación estándar de la intensidad del píxel.
        # Un mayor std dev implica más variación de color/intensidad, necesitando un sigma mayor.
        _, std_dev = cv2.meanStdDev(original_img)
        sigma_color = max(25, int(std_dev[0][0] * 1.5))

        # 3. Sigma espacio adaptativo, proporcional al diámetro.
        sigma_space = sigma_color

        # logger.debug(f"Parámetros adaptativos: d={d}, sigma_color={sigma_color}, sigma_space={sigma_space}")
        return use_bilateral_filter(original_img, d, sigma_space, sigma_color)
