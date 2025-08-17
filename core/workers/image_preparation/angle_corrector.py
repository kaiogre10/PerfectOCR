# PerfectOCR/core/workers/polygonal/angle_corrector.py
import cv2
import numpy as np
import logging
import math
from typing import Dict, Any
from core.factory.abstract_worker import ImagePrepAbstractWorker
from core.domain.data_formatter import DataFormatter

logger = logging.getLogger(__name__)

class AngleCorrector(ImagePrepAbstractWorker):
    """
    Worker especializado en detectar y corregir el ángulo de inclinación de una imagen.
    """
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.corrections = config
        logger.debug("AngleCorrector inicializado con Paddle.")

    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """
        Implementa el método abstracto de AbstractWorker.
        """
        
        # Intentamos obtener img_dims de varias fuentes posibles
        img_dims = context.get("img_dims", {})
        
        # A este punto, tenemos img_dims válido o hemos retornado False
        full_img = context.get("full_img")
        if full_img is None:
            logger.error("AngleCorrector: full_img no encontrado en contexto")
            return False
            
        # Debug para ver qué valores estamos usando
        logger.debug(f"AngleCorrector: Procesando imagen con dimensiones {img_dims}")
        
        full_img = self.correct(full_img, img_dims)
        
        # Actualiza la imagen en el contexto
        context["full_img"] = full_img
        
        # Y también en el manager si es posible
        try:
            dict_id = context.get("dict_id")
            if dict_id:
                manager.update_full_img(dict_id, full_img)
        except Exception as e:
            logger.warning(f"AngleCorrector: No se pudo actualizar full_img en el manager: {e}")
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
            logger.debug("No se detectaron líneas para la corrección de inclinación.")
            return full_img

        angles = [math.degrees(math.atan2(l[0][3]-l[0][1], l[0][2]-l[0][0])) for l in lines]
        filtered_angles = [a for a in angles if hough_angle_filter_range_degrees[0] < a < hough_angle_filter_range_degrees[1]]
        
        if not filtered_angles:
            logger.debug("Ninguna línea detectada en el rango de ángulos para corrección.")
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
