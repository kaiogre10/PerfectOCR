# PerfectOCR/core/workers/preprocessing/ink_enhancer.py
import cv2
import numpy as np
import logging
import time
from typing import Dict, Any, List
from core.factory.abstract_worker import PreprocessingAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons

logger = logging.getLogger(__name__)

class InkEnhancer(PreprocessingAbstractWorker):
    """Worker especializado en restaurar texto con tinta gastada o de baja intensidad."""

    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = self.config.get('ink_enhancement', {})
        self.faded_threshold = self.worker_config.get('faded_detection_threshold')
        self.contrast_boost = self.worker_config.get('contrast_boost_factor')
        self.enabled_outputs = self.config.get("enabled_outputs", {})
        self.output = self.enabled_outputs.get("ink_poly", False)

    def preprocess(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """Detecta y restaura texto con tinta gastada."""
        try:
            start_time = time.time()
            logger.debug("Mejoramiento de tinta empezado con éxito")
            if not manager.validate_cropped_img():
                logger.info(f"Sin cropped_img en el formatter")
                return False
                
            logger.debug("Polygonos revisados")
            
            polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
            if not polygons:
                return False

            # 1. Fase de Análisis
            analysis_results: List[Dict[str, Any]] = []
            poly_ids_order: List[str] = []

            for poly_id, polygon in polygons.items():
                cropped_img = polygon.cropped_img.cropped_img if polygon.cropped_img else None
                if cropped_img is None:
                    continue

                analysis = self._analyze_ink_quality(cropped_img)
                if analysis:
                    analysis_results.append(analysis)
                    poly_ids_order.append(poly_id)

            if not analysis_results:
                return True

            # 2. Fase de Decisión
            faded_scores = np.array([res['faded_score'] for res in analysis_results], dtype=np.float32)
            needs_enhancement = faded_scores > self.faded_threshold

            # 3. Fase de Aplicación
            enhanced_count = 0
            for idx, poly_id in enumerate(poly_ids_order):
                if not needs_enhancement[idx]:
                    continue

                polygon = polygons[poly_id]
                original_img = analysis_results[idx]['original_img']
                faded_score = faded_scores[idx]

                # logger.debug(f"Poly '{poly_id}': Restaurando tinta gastada (score: {faded_score:.2f})")

                enhanced_img = self._restore_faded_ink(original_img, faded_score)
                polygon.cropped_img.cropped_img = enhanced_img
                enhanced_count += 1

                if self.output:
                    from services.output_service import save_croped_image
                    worker_name = context.get("worker_name") or "inker"
                    image_name = manager.workflow.metadata.image_name if manager.workflow else ""
                    output_paths = context.get("output_paths", [])
                    
                    save_croped_image(image_name, poly_id, enhanced_img, output_paths, worker_name)
                    
            total_time = time.time() - start_time
            
            logger.debug(
                f"Restauración de tinta completada para {enhanced_count}/{len(poly_ids_order)} polígonos en: {total_time:.3f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Error en InkEnhancer: {e}", exc_info=True)
            return False

    def _analyze_ink_quality(self, img: np.ndarray[Any, np.dtype[np.uint8]]) -> Dict[str, Any]:
        """Analiza la calidad de la tinta y detecta si está gastada."""
        # Calcular estadísticas básicas
        mean_val = np.mean(img)
        std_val = np.std(img)

        # Detectar predominio de grises medios (característica de tinta gastada)
        mid_gray_ratio = np.sum((img >= 80) & (img <= 180)) / img.size

        # Calcular contraste local usando filtros
        kernel = np.ones((3, 3), np.float32) / 9
        local_mean = cv2.filter2D(img.astype(np.float32), -1, kernel)
        local_contrast = np.mean(np.abs(img.astype(np.float32) - local_mean))

        # Analizar histograma
        hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
        hist_normalized = hist / np.sum(hist)

        # El pico del histograma en grises medios indica tinta gastada
        mid_gray_peak = np.max(hist_normalized[80:180])

        # Calcular score de tinta gastada (0-1, donde 1 = muy gastada)
        faded_score = (
                (mid_gray_ratio * 0.4) +
                (min(1.0, (150 - mean_val) / 100) * 0.3) +  # Penalizar intensidades muy bajas
                (max(0, (40 - std_val) / 40) * 0.2) +  # Penalizar bajo contraste
                (mid_gray_peak * 0.1)
        )

        return {
            "original_img": img,
            "faded_score": min(1.0, faded_score),
            "mean_val": mean_val,
            "std_val": std_val,
            "mid_gray_ratio": mid_gray_ratio,
            "local_contrast": local_contrast
        }

    def _restore_faded_ink(self, img: np.ndarray[Any, np.dtype[np.uint8]], faded_score: float) -> np.ndarray[Any, np.dtype[np.uint8]]:
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

    def _save_debug_image(self, context: Dict[str, Any], poly_id: str, image: np.ndarray[Any, Any]):
        """Guarda una imagen de depuración."""
        from services.output_service import save_image
        import os

        output_paths = context.get("output_paths", [])
        for path in output_paths:
            output_dir = os.path.join(path, "inker")
            file_name = f"{poly_id}_inker_debug.png"
            save_image(image, output_dir, file_name)

        if output_paths:
            logger.debug(f"Imagen de debug de inker para '{poly_id}' guardada en {len(output_paths)} ubicaciones.")