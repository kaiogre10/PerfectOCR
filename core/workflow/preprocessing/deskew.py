# PerfectOCR/core/workflow/preprocessing/deskew.py
import os
import cv2
import numpy as np
import logging
import math
from typing import Dict, Any, List, Tuple
from paddleocr import PaddleOCR 

logger = logging.getLogger(__name__)

class Deskewer:
    """Detecta inclinación"""
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.corrections = config
        self.paddle_config = config.get('paddle', {})
        
        if not PaddleOCR:

            try:
                # Configuración directa y simple
                init_params = {
                    'use_angle_cls': self.paddle_config.get('use_angle_cls', False),
                    'lang': self.paddle_config.get('lang', 'es'),
                    'show_log': self.paddle_config.get('show_log', False),
                    'use_gpu': self.paddle_config.get('use_gpu', False),
                    'enable_mkldnn': self.paddle_config.get('enable_mkldnn', True)
                }

                for model_key in ['det_model_dir', 'rec_model_dir', 'cls_model_dir']:
                    model_path = self.paddle_config.get(model_key)
                    if model_path and os.path.exists(model_path):
                        init_params[model_key] = model_path
                
                self.engine = PaddleOCR(**init_params)
            except Exception as e:
                logger.error(f"Error crítico al inicializar PaddleOCR engine: {e}", exc_info=True)
                self.engine = None
        
        
    def _detect_angle(self, clean_img: np.ndarray, dpi_img: tuple) -> tuple[np.ndarray, int]:
        deskew_corrections = self.corrections
        canny = deskew_corrections.get('canny_thresholds', [50, 150])
        hough_thresh = deskew_corrections.get('hough_threshold', 150)
        max_gap = deskew_corrections.get('hough_max_line_gap_px', 20)
        angle_range = deskew_corrections.get('hough_angle_filter_range_degrees', [-15.0, 15.0])
        min_len_cap = deskew_corrections.get('hough_min_line_length_cap_px', 300)
        min_angle = deskew_corrections.get('min_angle_for_correction', 0.1)

        # Detección del ángulo usando OpenCV (método original)
        h, w = clean_img.shape[:2]
        img_dims = h, w
        center = (w // 2, h // 2)
        min_len = min(clean_img.shape[1] // 3, min_len_cap)
        edges = cv2.Canny(clean_img, canny[0], canny[1])
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=hough_thresh,
                                minLineLength=min_len, maxLineGap=max_gap)

        if lines is None or len(lines) == 0:
            return clean_img

        angles = [math.degrees(math.atan2(l[0][3]-l[0][1], l[0][2]-l[0][0])) for l in lines]
        filtered = [a for a in angles if angle_range[0] < a < angle_range[1]]
        angle = np.median(filtered) if filtered else 0.0
    
        # Aplicar corrección de ser necesario
        if abs(angle) > min_angle:
            logger.info(f"-> Aplicando corrección de inclinación: {angle:.2f} grados.")
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            deskewed_img = cv2.warpAffine(clean_img, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        else:
            deskewed_img = clean_img
            img_dims = {'height': h, 'width': w}

        return deskewed_img, img_dims
                
    def _detect_geometry(self, img_to_poly: np.ndarray, dpi_img, img_dims: Dict) -> tuple[List[List[List[float]]]]:
        """Detecta geometría usando solo el detector de PaddleOCR."""
        try:
            # Verifica si el motor de PaddleOCR está inicializado
            if self.engine is None:
                logger.error("PaddleOCR engine no está inicializado")  # Registra un error si no está inicializado
                return []  # Retorna una lista vacía si no hay engine
            
            # Ejecuta solo la detección de texto (sin reconocimiento), usando el detector de PaddleOCR
            results = self.engine.ocr(img_to_poly, cls=False, rec=False)
             
            # Verifica si no se detectaron resultados o si la lista está vacía
            if not results or not results[0]:
                logger.warning("No se detectaron regiones de texto")  # Advierte si no hay regiones detectadas
                return []  # Retorna una lista vacía
            
            # Inicializa una lista para almacenar los polígonos detectados
            polygons = []
            # Itera sobre cada detección encontrada en la imagen
            for detection in results[0]:
                # Si el objeto detection tiene el método 'tolist', lo convierte a lista, si no, lo deja igual
                polygon = detection.tolist() if hasattr(detection, 'tolist') else detection
                # Agrega el polígono a la lista de polígonos
                polygons.append(polygon)
            
            # Informa cuántos polígonos de texto fueron detectados
            logger.info(f"Detectados {len(polygons)} polígonos de texto")
            # Devuelve la lista de polígonos detectados
            return polygons
            
        except Exception as e:
            # Si ocurre cualquier excepción, la registra como error
            logger.error(f"Error en detección geométrica: {e}")
            # Retorna una lista vacía en caso de error
            return []
    
    def _get_polygons(self, clean_img: np.ndarray, dpi_img: int) -> tuple[np.ndarray, List[List[List[float]]], tuple[int, int], int]:
        """Retorna la imagen deskewed, coordenadas de polígonos, dimensiones y DPI."""
        # Primero aplicar deskew usando OpenCV
        deskewed_img, img_dims = self._detect_angle(clean_img, dpi_img)
        
        img_to_poly = deskewed_img.copy()

        # Detectar geometría para obtener las coordenadas de los polígonos
        polygons_coords = self._detect_geometry(img_to_poly, dpi_img, img_dims)
        logger.debug(f"Detectados {len(polygons_coords)} polígonos en imagen corregida")
        
        # Retornar la imagen corregida, coordenadas de polígonos, dimensiones y DPI
        return deskewed_img, polygons_coords, img_dims, dpi_img