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
        
        try:
            # Configuración directa y simple
            init_params = {
                "use_angle_cls": False,
                "lang": "es",
                "show_log": False,
                "use_gpu": False,
                "enable_mkldnn": True,
                "det_model_dir": r"c:\PerfectOCR\data\models\paddle\det\es",
                "rec_model_dir": r"c:\PerfectOCR\data\models\paddle\rec\es",
                "cls_model_dir": r"c:\PerfectOCR\data\models\paddle\cls"
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
        metadata = img_dims, dpi_img
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
            dpi_img = dpi_img
            img_dims = {'height': h, 'width': w}
            metadata = img_dims, dpi_img
    
        return deskewed_img, metadata
        
    def _detect_geometry(self, img_to_poly: np.ndarray, metadata: list[tuple]) -> List[List[List[float]]]:
        """Detecta geometría usando solo el detector de PaddleOCR."""
        # Cambiar temporalmente el nivel de log para debug
        old_level = logger.level
        logger.setLevel(logging.DEBUG)
        
        if not self.engine:
            logger.error("PaddleOCR engine no está inicializado")
            logger.setLevel(old_level)
            return [], metadata
        
        # Log información de la imagen de entrada
        img_shape = img_to_poly.shape
        img_dtype = img_to_poly.dtype
        img_min = img_to_poly.min()
        img_max = img_to_poly.max()
        logger.info(f"-> Imagen para detección: shape={img_shape}, min={img_min}, max={img_max}")
        
        try:
            results = self.engine.ocr(img_to_poly, cls=False, rec=False)
                        
            if results and len(results) > 0:
                logger.info(f"-> polígonos: {len(results[0]) if results[0] else 'None'}")
            
            if not results or not results[0]:
                logger.warning("No se detectaron regiones de texto")
                logger.setLevel(old_level)
                return [], metadata
                
        except Exception as e:
            logger.error(f"Error en PaddleOCR.ocr(): {e}", exc_info=True)
            logger.setLevel(old_level)
            return [], metadata
                
        polygons = []
        for line_counter, item_tuple in enumerate(results[0]):
            
            # Con rec=False, item_tuple son directamente las coordenadas del polígono
            if not isinstance(item_tuple, list) or len(item_tuple) < 3:
                if line_counter < 3:
                    logger.info(f"DEBUG: Item {line_counter} descartado: no es lista o menos de 3 puntos")
                continue
                
            bbox_polygon_raw = item_tuple
            
            try:
                bbox_polygon = [[float(p[0]), float(p[1])] for p in bbox_polygon_raw]
                if len(bbox_polygon) < 3:
                    if line_counter < 3:
                        logger.info(f"DEBUG: Polígono {line_counter} descartado: menos de 3 puntos ({len(bbox_polygon)})")
                    continue
                polygons.append(bbox_polygon)
            except (TypeError, ValueError, IndexError) as e:
                if line_counter < 3:
                    logger.info(f"DEBUG: Error procesando polígono {line_counter}: {e}")
                continue
                
        cantidad_poligonos = len(polygons)
        logger.info(f"-> Cantidad de polígonos encontrados: {cantidad_poligonos}")

        # Restaurar el nivel de log original
        logger.setLevel(old_level)
        return polygons, metadata

    
    def _get_polygons(self, clean_img: np.ndarray, dpi_img: int) -> tuple[np.ndarray, List[List[List[float]]], tuple[int, int], int]:
        """Retorna la imagen deskewed, coordenadas de polígonos, dimensiones y DPI."""
        # Primero aplicar deskew usando OpenCV
        deskewed_img, metadata = self._detect_angle(clean_img, dpi_img)
        
        img_to_poly = deskewed_img.copy()

        # Detectar geometría para obtener las coordenadas de los polígonos
        polygons, metadata = self._detect_geometry(img_to_poly, metadata)
        logger.debug(f"Detectados {len(polygons)} polígonos en imagen corregida")
                
        # Retornar la imagen corregida, coordenadas de polígonos, dimensiones y DPI
        return deskewed_img, polygons, metadata