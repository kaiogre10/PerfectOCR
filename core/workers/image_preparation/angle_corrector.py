# PerfectOCR/core/workers/image_preparation/angle_corrector.py
import cv2
import numpy as np
import logging
import math
from typing import Dict, Any, Optional
from core.factory.abstract_worker import ImagePrepAbstractWorker
from core.domain.data_formatter import DataFormatter

logger = logging.getLogger(__name__)

class AngleCorrector(ImagePrepAbstractWorker):
    """
    Worker especializado en detectar y corregir el ángulo de inclinación de una imagen.
    """
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get('angle_corrector', {})
        self.enabled_outputs = self.config.get("enabled_outputs", {})
        
    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        try:
            full_img = manager.workflow.full_img if manager.workflow else None
            if full_img is None:
                logger.error(f"No Hay full_img en el Formatter")
                return False
                
            logger.debug("Full_img obtenida con éxito")

            full_img = self.correct_angle(full_img, manager)
            
            manager.update_full_img(full_img)
            logger.debug(f"Imagen inclinada corregida actualiada en el manager")
                
            return True
            
        except Exception as e:
            logger.error(f"Error angular; {e}", exc_info=True)
            return False

    def correct_angle(self, full_img: np.ndarray[Any, np.dtype[np.uint8]], manager: DataFormatter) -> Optional[np.ndarray[Any, np.dtype[np.uint8]]]:
        """
        Aplica deskew a la imagen si es necesario y retorna la imagen (corregida o no).
        """
        min_angle_for_correction = self.worker_config.get('min_angle_for_correction')
        canny_thresholds = self.worker_config.get('canny_thresholds', [])
        hough_threshold = self.worker_config.get('hough_threshold')
        hough_max_line_gap_px = self.worker_config.get('hough_max_line_gap_px')
        hough_angle_filter_range_degrees = self.worker_config.get('hough_angle_filter_range_degrees', [])
        hough_min_line_length_cap_px = self.worker_config.get('hough_min_line_length_cap_px')
        
        try:
            # total_time = time.perf_counter()
            # img_dims: Dict[str, int] = {}
            # if manager.workflow and hasattr(manager.workflow, "metadata") and hasattr(manager.workflow.metadata, "img_dims"):
            #     img_dims = dict(getattr(manager.workflow.metadata, "img_dims", {}))
            
            img_dims: Dict[str, int] = manager.workflow.metadata.img_dims if manager.workflow else {}
                
            h = img_dims.get("height") 
            
            w = img_dims.get("width") 

            if h is None or w is None:
                logger.warning("Dimensiones de imagen inválidas (None o 0) para la corrección de ángulo.")
                return full_img

            center = w // 2, h // 2
            min_len = min((w) // 3, hough_min_line_length_cap_px)
            
            edges = cv2.Canny(full_img, canny_thresholds[0], canny_thresholds[1])
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=hough_threshold,
                                    minLineLength=min_len, maxLineGap=hough_max_line_gap_px)

            if lines is None or len(lines) == 0:
                # logger.debug(f"No se detectaron líneas para la corrección de inclinación, {time.perf_counter() - total_time:.6f}s")
                return full_img

            angles = [math.degrees(math.atan2(l[0][3]-l[0][1], l[0][2]-l[0][0])) for l in lines]
            filtered_angles = [a for a in angles if hough_angle_filter_range_degrees[0] < a < hough_angle_filter_range_degrees[1]]
            
            if not filtered_angles:
                # logger.debug(f"Ninguna línea detectada en el rango de ángulos para corrección, tiempo: {time.perf_counter() - total_time:.6f}s")
                return full_img

            angle = np.median(filtered_angles)
            if abs(angle) > min_angle_for_correction:
                logger.debug(f"-> Aplicando corrección de inclinación: {angle:.6f} grados.")
                rotation_matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
                deskewed_img = cv2.warpAffine(full_img, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                full_img = np.array(deskewed_img, dtype=np.uint8)
                # logger.debug(f"Imagen rotada en {time.perf_counter() - total_time:.6f}s")
                return full_img
            else:             
                # logger.debug(f"Ángulo de inclinación insignificante. No se aplica corrección, tiempo de medición: {time.perf_counter() - total_time:.6f}s")
                return full_img
                
        except Exception as e:
            logger.error(f"ERRROR; {e}", exc_info=True)
            return full_img
            
#     def _trim_using_hough(self, full_img: np.ndarray[Any, Any], lines: np.ndarray[Any, Any], img_dims: Dict[str, int]) -> np.ndarray[Any, Any]:
#         """Trim usando líneas de Hough detectadas"""
        
#         h = img_dims.get("height")
#         w = img_dims.get("width")
        
#         # Encontrar extremos de todas las líneas
#         min_x, max_x = w, 0
#         min_y, max_y = h, 0
        
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             min_x = min(min_x, x1, x2)
#             max_x = max(max_x, x1, x2)
#             min_y = min(min_y, y1, y2)
#             max_y = max(max_y, y1, y2)
        
#         # Margen de seguridad
#         margin = 10

#         # Asegurarse de que min_x, min_y, max_x, max_y no sean None y sean valores válidos
#         if min_x is None or min_y is None or max_x is None or max_y is None:
#             logger.warning("No se pudo determinar los extremos de las líneas de Hough para recorte. Se devuelve la imagen completa.")
#             return full_img

#         x1 = max(0, int(min_x) - margin)
#         y1 = max(0, int(min_y) - margin)
#         x2 = min(int(w), int(max_x) + margin)
#         y2 = min(int(h), int(max_y) + margin)

#         logger.debug(f"Recorte usando Hough: x1={x1}, y1={y1}, x2={x2}, y2={y2} (w={w}, h={h})")
#         # ...existing code...
#         if x1 == 0 and y1 == 0 and x2 == w and y2 == h:
#             logger.debug("El recorte coincide con la imagen completa. No se realiza recorte adicional.")
#         else:
#             logger.debug("Se detectó región a recortar usando líneas de Hough.")

#         # Log de las nuevas dimensiones
#         new_height = y2 - y1
#         new_width = x2 - x1
#         logger.debug(f"Nuevas dimensiones tras recorte: width={new_width}, height={new_height}")

#         return full_img[y1:y2, x1:x2]
# #