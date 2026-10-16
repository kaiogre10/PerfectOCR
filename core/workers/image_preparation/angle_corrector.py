# PerfectOCR/core/workers/image_preparation/angle_corrector.py
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
from core.contracts.abstract_worker import ImagePrepAbstractWorker
from domain.data_formatter import DataFormatter
from utils.image_utils import get_image_lines
from components.compiled.image import rotate

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
            new_ptr, corrected = self.correct_angle(manager)

            logger.debug(f"Imagen rotada actuallizada con éxito.")
            if corrected and new_ptr:
                if manager.update_image(new_ptr):
                    logger.debug(f"Imagen rotada con éxito.")
                    return True
                else:
                    return False
            
        except Exception as e:
            logger.error(f"Error angular; {e}", exc_info=True)
            return False

        return True

    def correct_angle(self, manager: DataFormatter) -> Tuple[Optional[int], bool]:
        """Aplica deskew a la imagen si es necesario retornando el puntero de la imagen corregida)"""
        corrected = False
        try:
            full_img = manager.get_full_img()
            if full_img is None:
                logger.error(f"No Hay full_img en el Formatter")
                return None, corrected

            h = full_img.shape[0]
            w = full_img.shape[1]

            min_len = min(w / 3.0, self.hough_min_line_length_cap_px)
            
            lines = get_image_lines(full_img, self.canny_thresholds, self.hough_threshold, min_len, self.hough_max_line_gap_px)
                
            if lines is None or len(lines) == 0: # type: ignore
               # logger.warning(f"No se detectaron líneas para la corrección de inclinación")
                return None, corrected

            angles = np.degrees(np.arctan2(lines[:, 0, 3] - lines[:, 0, 1], lines[:, 0, 2] - lines[:, 0, 0]))
            filtered_angles = angles[(angles > self.hough_angle_filter_range_degrees[0]) & (angles < self.hough_angle_filter_range_degrees[1])]
            if filtered_angles.size == 0:
               # logger.warning(f"Ninguna línea detectada en el rango de ángulos para corrección")
                return None, corrected

            angle = np.median(filtered_angles)
            if np.abs(angle) > self.min_angle_for_correction:
                img_ptr = manager.get_img_ptr()
                if not img_ptr:
                    logger.error(f"PUNTERO INEXISTENTE: {img_ptr}")
                    return None, corrected

                center = (w / 2.0, h / 2.0)
                
                new_ptr = rotate(img_ptr, angle, center[0], center[1]) # type: ignore
                if not new_ptr:
                    logger.error(f"PUNTERO ROTADO CON ERROR: {img_ptr}")
                    return None, corrected

                return new_ptr, True
            
            logger.debug(f"Ángulo de inclinación '{angle}°' insignificante. No se aplica corrección")
            return None, corrected
                
        except Exception as e:
            logger.error(f"ERRROR; {e}", exc_info=True)
            return None, corrected

    # def rotate_and_crop(self, img: np.ndarray[Any, np.dtype[np.uint8]]) -> np.ndarray[Any, np.dtype[np.uint8]]:
    #     try:
    #         # Si la imagen es RGB, conviértela a escala de grises para el recorte
    #         if img.ndim == 3:
    #             mask = img < self.border_cutt
    #             #  = np.where(cond, img)
    #         else:
    #             mask = img < self.border_cutt

    #         coords = np.column_stack(np.where(~mask))
    #         if coords.size == 0:
    #             return img  # No hay contenido, retorna original

    #         y_min = coords[:, 0].min()
    #         y_max = coords[:, 0].max()
    #         x_min = coords[:, 1].min()
    #         x_max = coords[:, 1].max()

    #         return img[y_min:y_max+1, x_min:x_max+1].astype(np.uint8)
    #     except Exception as e:
    #         logger.info(f"Error recortando: {e}", exc_info=True)
    #     return img
