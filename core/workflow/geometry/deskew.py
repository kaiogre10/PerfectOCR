# PerfectOCR/core/workflow/geometry/deskew.py
import os
import cv2
import numpy as np
import logging
import math
from typing import Dict, Any, List, Tuple, Union

from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)

class Deskewer:
    """
    Detecta y corrige la inclinación de la página y extrae la geometría
    del texto usando una instancia LIGERA de PaddleOCR (solo detección).
    """
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.corrections = config
        self.paddle_config = config.get('paddle', {})
        self.engine = None

        logger.info("Inicializando instancia de PaddleOCR para GEOMETRÍA (solo detección)...")
        
        try:
            # Forzamos rec=False y use_angle_cls=False para cargar solo el detector.
            init_params = {
                "use_angle_cls": False,
                "rec": False,  # <- ¡CLAVE! No carga el modelo de reconocimiento.
                "lang": self.paddle_config.get('lang', 'es'),
                "show_log": self.paddle_config.get('show_log', False),
                "use_gpu": self.paddle_config.get('use_gpu', False),
                "enable_mkldnn": self.paddle_config.get('enable_mkldnn', True)
            }
            
            # Carga de modelos locales, priorizando el de detección.
            logger.debug("Buscando modelo de detección local de PaddleOCR...")
            det_model_path = self.paddle_config.get('det_model_dir')
            if det_model_path and os.path.exists(det_model_path):
                init_params['det_model_dir'] = det_model_path
                logger.info(f"Detector local encontrado y añadido: {det_model_path}")
            else:
                logger.warning("No se encontró un directorio de modelo de detección local. PaddleOCR intentará descargarlo.")

            logger.debug(f"Parámetros finales para inicializar PaddleOCR (geometría): {init_params}")
            self.engine = PaddleOCR(**init_params)
            logger.info("Instancia de PaddleOCR para GEOMETRÍA inicializada exitosamente.")

        except Exception as e:
            logger.error(f"Error crítico al inicializar la instancia geométrica de PaddleOCR: {e}", exc_info=True)

    def _detect_angle(self, clean_img: np.ndarray, dpi_img: int) -> Tuple[np.ndarray, Tuple[Dict[str, int], int]]:
        deskew_corrections = self.corrections
        canny = deskew_corrections.get('canny_thresholds', [50, 150])
        hough_thresh = deskew_corrections.get('hough_threshold', 150)
        max_gap = deskew_corrections.get('hough_max_line_gap_px', 20)
        angle_range = deskew_corrections.get('hough_angle_filter_range_degrees', [-15.0, 15.0])
        min_len_cap = deskew_corrections.get('hough_min_line_length_cap_px', 300)
        min_angle = deskew_corrections.get('min_angle_for_correction', 0.1)

        h, w = clean_img.shape[:2]
        center = (w // 2, h // 2)
        min_len = min(clean_img.shape[1] // 3, min_len_cap)
        edges = cv2.Canny(clean_img, canny[0], canny[1])
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=hough_thresh,
                                minLineLength=min_len, maxLineGap=max_gap)

        if lines is None or len(lines) == 0:
            metadata = ({'height': h, 'width': w}, dpi_img)
            return clean_img, metadata

        angles = [math.degrees(math.atan2(l[0][3]-l[0][1], l[0][2]-l[0][0])) for l in lines]
        filtered = [a for a in angles if angle_range[0] < a < angle_range[1]]
        angle = np.median(filtered) if filtered else 0.0
    
        if abs(angle) > min_angle:
            logger.info(f"-> Aplicando corrección de inclinación: {angle:.2f} grados.")
            rotation_matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
            deskewed_img = cv2.warpAffine(clean_img, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        else:
            deskewed_img = clean_img
        
        final_h, final_w = deskewed_img.shape[:2]
        metadata = ({'height': final_h, 'width': final_w}, dpi_img)
        return deskewed_img, metadata
        
    def _detect_geometry(self, img_to_poly: np.ndarray, metadata: Tuple) -> Tuple[List[List[List[float]]], Tuple]:
        """Detecta geometría usando solo el detector de PaddleOCR."""
        # Cambiar temporalmente el nivel de log para debug
        old_level = logger.level
        logger.setLevel(logging.DEBUG)
        
        if not self.engine:
            logger.error("El motor de geometría de PaddleOCR no está inicializado.")
            logger.setLevel(old_level)
            return [], metadata
        
        logger.info(f"-> Ejecutando detección geométrica en imagen de {img_to_poly.shape}...")
        
        try:
            # Llamada explícita para solo detección (rec=False)
            results = self.engine.ocr(img_to_poly, cls=False, rec=False)
            
            if results and len(results) > 0 and results[0] is not None:
                logger.info(f"-> Detección geométrica exitosa. Polígonos crudos encontrados: {len(results[0])}")
            else:
                logger.warning("La detección geométrica no encontró polígonos de texto.")
                logger.setLevel(old_level)
                return [], metadata
                
        except Exception as e:
            logger.error(f"Error durante la detección geométrica con PaddleOCR: {e}", exc_info=True)
            logger.setLevel(old_level)
            return [], metadata
                
        polygons = []
        for bbox_polygon_raw in results[0]:
            if not isinstance(bbox_polygon_raw, list) or len(bbox_polygon_raw) < 3:
                continue
            
            try:
                # Asegurar que las coordenadas son flotantes
                polygon_coords = [[float(p[0]), float(p[1])] for p in bbox_polygon_raw]
                polygons.append(polygon_coords)
            except (TypeError, ValueError, IndexError):
                continue
                
        logger.info(f"-> Polígonos válidos procesados: {len(polygons)}")

        # Restaurar el nivel de log original
        logger.setLevel(old_level)
        return polygons, metadata

    def _get_polygons(self, clean_img: np.ndarray, dpi_img: int) -> Tuple[np.ndarray, List[List[List[float]]], Tuple]:
        """Retorna la imagen deskewed, coordenadas de polígonos y la metadata original."""
        # Primero aplicar deskew usando OpenCV
        deskewed_img, metadata = self._detect_angle(clean_img, dpi_img)
        
        # Copiamos para no modificar la imagen original que podría ser usada en otro lugar
        img_for_geometry = deskewed_img.copy()

        # Detectar geometría para obtener las coordenadas de los polígonos
        polygons, metadata = self._detect_geometry(img_for_geometry, metadata)
        logger.debug(f"Detectados {len(polygons)} polígonos en la fase de geometría.")
                
        # Retornar la imagen corregida, coordenadas de polígonos y la metadata
        return deskewed_img, polygons, metadata
        