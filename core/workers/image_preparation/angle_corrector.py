# PerfectOCR/core/workers/image_preparation/angle_corrector.py
import numpy as np
import logging
from typing import Dict, Any, Tuple
from core.contracts.abstract_worker import ImagePrepAbstractWorker
from domain.data_formatter import DataFormatter
from utils.image_utils import get_rotation_matrix, get_image_lines, rotate_matrix
from services.output_service import save_croped_image

logger = logging.getLogger(__name__)

class AngleCorrector(ImagePrepAbstractWorker):
    """Worker especializado en detectar y corregir el ángulo de inclinación de una imagen"""
    __slots__ = (
        "min_angle",
        "canny_thresholds",
        "hough_threshold",
        "hough_max_line_gap_px",
        "hough_angle_filter_range_degrees",
        "hough_min_line_length_cap_px",
        "output"
    )
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        worker_config = config.get("angle_corrector", {})
        self.min_angle_for_correction = worker_config.get('min_angle_for_correction')
        self.canny_thresholds = worker_config['canny_thresholds']
        self.hough_threshold = worker_config.get('hough_threshold')
        self.hough_max_line_gap_px = worker_config.get('hough_max_line_gap_px')
        self.hough_angle_filter_range_degrees = worker_config['hough_angle_filter_range_degrees']
        self.hough_min_line_length_cap_px = worker_config.get('hough_min_line_length_cap_px')
        self.output = config.get("angle_corrected", False)
        
    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        try:
            full_image = manager.get_full_img()
            if full_image is None:
                logger.error(f"No Hay full_img en el Formatter")
                return False
            
            logger.debug("Full_img obtenida con éxito")
            
            # full_image = make_contiguous(full_image)

            full_img, corrected = self.correct_angle(full_image)

            # if manager.update_full_img(corrected, full_img):
            logger.debug(f"Imagen rotada actuallizada con éxito.")

            if self.output and corrected:
                image_name = manager.workflow.metadata.image_name if manager.workflow else ""
                img_id = f"full_img_{image_name}"
                save_croped_image(image_name, img_id, full_img)
            return True
            
        except Exception as e:
            logger.error(f"Error angular; {e}", exc_info=True)
        return True

    def correct_angle(self, full_img: np.ndarray[Any, np.dtype[np.uint8]]) -> Tuple[np.ndarray[Any, np.dtype[np.uint8]], bool]:
        """Aplica deskew a la imagen si es necesario y retorna la imagen (corregida o no)."""
        try:
            h =  full_img.shape[0]
            w =  full_img.shape[1]
            
            center = w // 2, h // 2
            min_len = min(w // 3, self.hough_min_line_length_cap_px)
            
            lines = get_image_lines(full_img, self.canny_thresholds, self.hough_threshold, min_len, self.hough_max_line_gap_px)
                
            if lines is None or len(lines) == 0: # type: ignore
               # logger.warning(f"No se detectaron líneas para la corrección de inclinación")
                return full_img, False

            angles = np.degrees(np.arctan2(lines[:, 0, 3] - lines[:, 0, 1], lines[:, 0, 2] - lines[:, 0, 0]))
            filtered_angles = angles[(angles > self.hough_angle_filter_range_degrees[0]) & (angles < self.hough_angle_filter_range_degrees[1])]
            if filtered_angles.size == 0:
               # logger.warning(f"Ninguna línea detectada en el rango de ángulos para corrección")
                return full_img, False

            angle = np.median(filtered_angles)
            if np.abs(angle) > self.min_angle_for_correction:
                rotation_matrix = get_rotation_matrix(center, angle)
            
            # Calcular nuevas dimensiones
                cos = np.abs(rotation_matrix[0, 0])
                sin = np.abs(rotation_matrix[0, 1])
                new_w = int((h * sin) + (w * cos))
                new_h = int((h * cos) + (w * sin))
                
                # Ajustar traslación para centrar
                rotation_matrix[0, 2] += (new_w / 2) - center[0]
                rotation_matrix[1, 2] += (new_h / 2) - center[1]
                
                logger.debug(f"Imagen rotada '{angle:.4f}°' ángulos")
                return rotate_matrix(full_img, rotation_matrix, new_w, new_h), True
            else:             
                logger.debug(f"Ángulo de inclinación '{angle}°' insignificante. No se aplica corrección")
                return full_img, False
                
        except Exception as e:
            logger.error(f"ERRROR; {e}", exc_info=True)
        return full_img, False

    def rotate_and_crop(self, img: np.ndarray[Any, np.dtype[np.uint8]]) -> np.ndarray[Any, np.dtype[np.uint8]]:
        try:
            # Si la imagen es RGB, conviértela a escala de grises para el recorte
            if img.ndim == 3:
                mask = img < self.border_cutt
                #  = np.where(cond, img)
            else:
                mask = img < self.border_cutt

            coords = np.column_stack(np.where(~mask))
            if coords.size == 0:
                return img  # No hay contenido, retorna original

            y_min = coords[:, 0].min()
            y_max = coords[:, 0].max()
            x_min = coords[:, 1].min()
            x_max = coords[:, 1].max()

            return img[y_min:y_max+1, x_min:x_max+1].astype(np.uint8)
        except Exception as e:
            logger.info(f"Error recortando: {e}", exc_info=True)
        return img
