# PerfectOCR/core/workers/polygonal/angle_corrector.py
import cv2
import numpy as np
import logging
import math
from typing import Dict, Any
from core.factory.abstract_worker import AbstractWorker
from core.domain.data_formatter import DataFormatter

logger = logging.getLogger(__name__)

class AngleCorrector(AbstractWorker):
    """
    Worker especializado en detectar y corregir el ángulo de inclinación de una imagen.
    """
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.corrections = config
        logger.info("AngleCorrector inicializado.")

    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """
        Implementa el método abstracto de AbstractWorker.
        """
        
        # Obtenemos los metadatos completos para acceder a img_dims
        metadata = context.get("metadata", {})
        
        # Intentamos obtener img_dims de varias fuentes posibles
        img_dims = context.get("img_dims", {})
        
        
        
        # Si img_dims no está directamente en el contexto, buscamos en metadata
        if not img_dims or not isinstance(img_dims, Dict) or "width" not in img_dims or "height" not in img_dims:
            if isinstance(metadata, Dict):
                img_dims = metadata.get("img_dims", {})
            
        # Si aún no tenemos img_dims, intentamos obtenerlo del DataFormatter
        if not img_dims or not isinstance(img_dims, Dict) or "width" not in img_dims or "height" not in img_dims:
            try:
                manager_metadata = manager.get_metadata()
                img_dims = manager_metadata.get("img_dims", {})
            except Exception as e:
                logger.warning(f"AngleCorrector: No se pudo obtener img_dims del manager: {e}")
        
        # Si no encontramos img_dims válidos en ninguna parte, usamos dimensiones predeterminadas
        if not img_dims or not isinstance(img_dims, Dict) or "width" not in img_dims or "height" not in img_dims:
            logger.warning("AngleCorrector: No se encontró img_dims válido, usando valores predeterminados")
            # Calculamos dimensiones a partir de la imagen si es posible
            full_img = context.get("full_img")
            if full_img is not None and hasattr(full_img, "shape"):
                h, w = full_img.shape[:2]
                img_dims = {"width": w, "height": h}
            else:
                # Usamos valores predeterminados
                img_dims = {"width": 1, "height": 1}
                logger.error("AngleCorrector: No se pudo determinar img_dims. Usando valores predeterminados.")
                return False
        
        # A este punto, tenemos img_dims válido o hemos retornado False
        full_img = context.get("full_img")
        if full_img is None:
            logger.error("AngleCorrector: full_img no encontrado en contexto")
            return False
            
        # Debug para ver qué valores estamos usando
        logger.info(f"AngleCorrector: Procesando imagen con dimensiones {img_dims}")
        
        full_img = self.correct(full_img, img_dims)
        
        # Actualiza la imagen en el contexto
        context["full_img"] = full_img
        
        # Y también en el manager si es posible
        try:
            dict_id = context.get("dict_id")
            if dict_id:
                manager._update_full_img(dict_id, full_img)
        except Exception as e:
            logger.warning(f"AngleCorrector: No se pudo actualizar full_img en el manager: {e}")
            # Pero continuamos porque la imagen está actualizada en el contexto
        
        return True


    def correct(self, full_img: np.ndarray[Any, Any], img_dims: Dict[str, int]) -> np.ndarray[Any, Any]:
        """
        Aplica deskew a la imagen si es necesario y retorna la imagen (corregida o no).
        """
        canny_thresholds = self.corrections.get('canny_thresholds', [50, 150])
        hough_threshold = self.corrections.get('hough_threshold', 150)
        hough_max_line_gap_px = self.corrections.get('hough_max_line_gap_px', 20)
        hough_angle_filter_range_degrees = self.corrections.get('hough_angle_filter_range_degrees', [-15.0, 15.0])
        hough_min_line_length_cap_px = self.corrections.get('hough_min_line_length_cap_px', 300)
        min_angle_for_correction = self.corrections.get('min_angle_for_correction', 0.1)

        h = img_dims.get("height")
        w = img_dims.get("width")

        # Ensure h and w are valid integers and not None
        if h is None or w is None or h == 0 or w == 0:
            logger.warning("Dimensiones de imagen inválidas (None o 0) para la corrección de ángulo.")
            return full_img

        center = w // 2, h // 2
        min_len = min((w) // 3, hough_min_line_length_cap_px)
        
        edges = cv2.Canny(full_img, canny_thresholds[0], canny_thresholds[1])
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=hough_threshold,
                                minLineLength=min_len, maxLineGap=hough_max_line_gap_px)

        if lines is None or len(lines) == 0:
            logger.info("No se detectaron líneas para la corrección de inclinación.")
            return full_img

        angles = [math.degrees(math.atan2(l[0][3]-l[0][1], l[0][2]-l[0][0])) for l in lines]
        filtered_angles = [a for a in angles if hough_angle_filter_range_degrees[0] < a < hough_angle_filter_range_degrees[1]]
        
        if not filtered_angles:
            logger.info("Ninguna línea detectada en el rango de ángulos para corrección.")
            return full_img

        angle = np.median(filtered_angles)

        if abs(angle) > min_angle_for_correction:
            logger.info(f"-> Aplicando corrección de inclinación: {angle:.2f} grados.")
            rotation_matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
            deskewed_img = cv2.warpAffine(full_img, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return deskewed_img
        else:
            logger.info("Ángulo de inclinación insignificante. No se aplica corrección.")
        return full_img
