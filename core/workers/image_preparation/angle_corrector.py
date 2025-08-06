# PerfectOCR/core/workers/polygonal/angle_corrector.py
import cv2
import numpy as np
import logging
import math
from typing import Dict, Any
from core.workers.factory.abstract_worker import AbstractWorker
from core.domain.workflow_job import ProcessingStage

logger = logging.getLogger(__name__)

class AngleCorrector(AbstractWorker):
    """
    Worker especializado en detectar y corregir el ángulo de inclinación de una imagen.
    """
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.corrections = config
        logger.info("AngleCorrector inicializado.")

    def process(self, image: np.ndarray[Any, Any], context: Dict[str, Any]) -> np.ndarray[Any, Any]:
        """
        Implementa el método abstracto de AbstractWorker.
        """
        workflow_job = context.get('workflow_job')
        
        # Obtener dimensiones de la imagen
        img_dims = {
            "width": image.shape[1],
            "height": image.shape[0]
        }
        
        # Aplicar corrección de ángulo
        corrected_image = self.correct(image, img_dims)
        
        # Actualizar el WorkflowJob si está disponible
        if workflow_job and workflow_job.full_img is not None:
            workflow_job.full_img = corrected_image
            workflow_job.update_stage(ProcessingStage.GEOMETRY_DETECTED)
        
        return corrected_image

    def correct(self, clean_img: np.ndarray[Any, Any], img_dims: Dict[str, Any]) -> np.ndarray[Any, Any]:
        """
        Aplica deskew a la imagen si es necesario y retorna la imagen (corregida o no).
        """
        canny_thresholds = self.corrections.get('canny_thresholds', [50, 150])
        hough_threshold = self.corrections.get('hough_threshold', 150)
        hough_max_line_gap_px = self.corrections.get('hough_max_line_gap_px', 20)
        hough_angle_filter_range_degrees = self.corrections.get('hough_angle_filter_range_degrees', [-15.0, 15.0])
        hough_min_line_length_cap_px = self.corrections.get('hough_min_line_length_cap_px', 300)
        min_angle_for_correction = self.corrections.get('min_angle_for_correction', 0.1)

        h = int(img_dims.get("height", 0) or 0)
        w = int(img_dims.get("width", 0) or 0)
        
        if h == 0 or w == 0:
            logger.warning("Dimensiones de imagen inválidas (0) para la corrección de ángulo.")
            return clean_img

        center = (w // 2, h // 2)
        min_len = min(w // 3, hough_min_line_length_cap_px)
        
        edges = cv2.Canny(clean_img, canny_thresholds[0], canny_thresholds[1])
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=hough_threshold,
                                minLineLength=min_len, maxLineGap=hough_max_line_gap_px)

        if lines is None or len(lines) == 0:
            logger.info("No se detectaron líneas para la corrección de inclinación.")
            return clean_img

        angles = [math.degrees(math.atan2(l[0][3]-l[0][1], l[0][2]-l[0][0])) for l in lines]
        filtered_angles = [a for a in angles if hough_angle_filter_range_degrees[0] < a < hough_angle_filter_range_degrees[1]]
        
        if not filtered_angles:
            logger.info("Ninguna línea detectada en el rango de ángulos para corrección.")
            return clean_img

        angle = np.median(filtered_angles)

        if abs(angle) > min_angle_for_correction:
            logger.info(f"-> Aplicando corrección de inclinación: {angle:.2f} grados.")
            rotation_matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
            deskewed_img = cv2.warpAffine(clean_img, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return deskewed_img
        else:
            logger.info("Ángulo de inclinación insignificante. No se aplica corrección.")
            return clean_img
