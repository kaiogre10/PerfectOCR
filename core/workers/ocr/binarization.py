# PerfectOCR/core/workers/ocr/interceptor.py
import cv2
import time
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from skimage.filters import threshold_sauvola  # type: ignore
from skimage.util import img_as_ubyte # type: ignore
from core.factory.abstract_worker import OCRAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons

logger = logging.getLogger(__name__)

class Binarizator(OCRAbstractWorker):
    """
    Actúa como un centro de decisión para la fragmentación de polígonos.
    - Binariza las imágenes de los polígonos para su análisis.
    - Realiza un análisis visual para detectar agrupaciones incorrectas.
    - Combina su análisis visual con las sugerencias basadas en texto (del TextCleaner)
      para crear una lista definitiva de polígonos que necesitan ser fragmentados.
    """
    
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.config = config
        bin_config = self.config.get('binarize', {})
        self.c_value = bin_config.get('c_value', 7)
        self.height_thresholds = bin_config.get('height_thresholds_px', [100, 800, 1500, 2500])
        self.block_sizes_map = bin_config.get('block_sizes_map', [15, 21, 25, 35, 41])

        # frag_config = self.config.get('fragmentation', {})
        # self.min_area_factor = frag_config.get('min_area_factor', 0.01)
        # self.min_contours_for_frag = frag_config.get('min_contours_for_frag', 3)
        # self.approx_poly_epsilon = frag_config.get('approx_poly_epsilon', 0.02)

    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """Binarización y extracción ligera de contornos -> guardamos solo boxes."""
        try:
            start = time.time()
            polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
            if not polygons:
                return False

            contours_meta: Dict[str, Dict[str, Any]] = {}

            for poly_id, polygon in polygons.items():
                cropped_img = polygon.cropped_img.cropped_img if polygon.cropped_img else None
                if cropped_img is None or cropped_img.size == 0:
                    continue

                height = int(cropped_img.shape[0])
                block = self._get_adaptive_block_size(height)
                mode = self._measure_polygon_quality(cropped_img)

                if mode == "otsu":
                    bin_img = self._otsu_binarize(cropped_img)
                elif mode == "adaptive_gaussian":
                    bin_img = self._adaptive_binarize(cropped_img, block)
                elif mode == "sauvola":
                    bin_img = self._sauvola_binarize(cropped_img, block)
                else:
                    bin_img = self._adaptive_mean_fallback(cropped_img, block)

                # Contornos y filtrado por área relativa
                contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue

                poly_area = float(bin_img.shape[0] * bin_img.shape[1])
                min_area = max(1.0, poly_area * getattr(self, "min_area_factor", 0.005))
                valid_boxes_norm: List[List[float]] = []

                for c in contours:
                    a = cv2.contourArea(c)
                    if a < min_area:
                        continue
                    x, y, w, h = cv2.boundingRect(c)
                    x2, y2 = x + w, y + h
                    valid_boxes_norm.append([x / bin_img.shape[1], y / bin_img.shape[0], x2 / bin_img.shape[1], y2 / bin_img.shape[0]])

                if valid_boxes_norm:
                    contours_meta[poly_id] = {
                        "contour_boxes_norm": valid_boxes_norm
                    }

            if contours_meta:
                # Guardar solo metadatos ligeros de contorno en el manager (sin imágenes)
                context['contours_meta'] = contours_meta

            logger.info(f"Binarizator: contornos extraídos para {len(contours_meta)} polígonos en {time.time()-start:.3f}s")
            return True

        except Exception as e:
            logger.error(f"Binarizator.transcribe error: {e}", exc_info=True)
            return False
        
    def _get_adaptive_block_size(self, height: float) -> int:
        for i, threshold in enumerate(self.height_thresholds):
            if height <= threshold:
                block_size = self.block_sizes_map[min(i, len(self.block_sizes_map) - 1)]
                return max(3, block_size if block_size % 2 != 0 else block_size + 1)
        final_block_size = self.block_sizes_map[-1]
        return max(3, final_block_size if final_block_size % 2 != 0 else final_block_size + 1)

    def _measure_polygon_quality(self, cropped_img: np.ndarray[Any, np.dtype[np.uint8]]) -> List[str]:
        std = np.std(cropped_img)
        if std == 0: return "adaptive_mean" # Imagen plana
        
        hist = cv2.calcHist([cropped_img], [0], None, [256], [0, 256]).flatten()
        peaks = np.sum((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:]))
        prob = hist / np.sum(hist)
        entropy = -np.sum(prob * np.log2(prob + 1e-8))

        if peaks >= 2 and std > 30: return "otsu"
        elif std > 20 and entropy > 4.5: return "adaptive_gaussian"
        elif std > 10: return "sauvola"
        else: return "adaptive_mean"
        
    def _otsu_binarize(self, cropped_img: np.ndarray[Any, np.dtype[np.uint8]]) -> np.ndarray[np.uint8, Any]:
        _, bin_img = cv2.threshold(cropped_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return bin_img
       
    def _adaptive_binarize(self, cropped_img: np.ndarray[Any, np.dtype[np.uint8]], block_size: int) -> np.ndarray[np.uint8, Any]:
        return cv2.adaptiveThreshold(
            cropped_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, block_size, self.c_value
        )
    
    def _sauvola_binarize(self, cropped_img: np.ndarray[Any, np.dtype[np.uint8]], adaptive_block_size: int) -> np.ndarray[np.uint8, Any]:
        thresh_sauvola = threshold_sauvola(cropped_img, window_size=adaptive_block_size)
        bin_img = (cropped_img > thresh_sauvola).astype(np.uint8) * 255
        return bin_img

    def _adaptive_mean_fallback(self, cropped_img: np.ndarray[Any, np.dtype[np.uint8]], block_size: int) -> np.ndarray[np.uint8, Any]:
        return cv2.adaptiveThreshold(
            cropped_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, block_size, max(1, self.c_value - 2)
        )

    def _visual_analysis_needs_fragmentation(self, bin_img: np.ndarray[np.uint8, Any]) -> Tuple[bool, List[Any]]:
        """Determina si la imagen binarizada parece contener múltiples elementos separados.
        Devuelve un bool indicando necesidad de fragmentar y la lista de contornos válidos."""
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, []

        # Filtrar contornos muy pequeños que son probablemente ruido
        poly_area = bin_img.shape[0] * bin_img.shape[1]
        min_area = poly_area * self.min_area_factor
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]

        return (len(valid_contours) >= self.min_contours_for_frag), valid_contours
