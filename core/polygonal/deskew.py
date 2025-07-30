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
            init_params = {
                "use_angle_cls": False,
                "rec": False, 
                "lang": self.paddle_config.get('lang', 'es'),
                "show_log": self.paddle_config.get('show_log', False),
                "use_gpu": self.paddle_config.get('use_gpu', False),
                "enable_mkldnn": self.paddle_config.get('enable_mkldnn', True)
            }
            
            det_model_path = self.paddle_config.get('det_model_dir')
            if det_model_path and os.path.exists(det_model_path):
                init_params['det_model_dir'] = det_model_path
            else:
                logger.warning("No se encontró un directorio de modelo de detección local. PaddleOCR intentará descargarlo.")
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
            dpi_img = dpi_img
            doc_metadata = ({'height': h, 'width': w}, dpi_img)
            return clean_img, doc_metadata

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
        doc_metadata = ({'height': final_h, 'width': final_w}, dpi_img)
        return deskewed_img, doc_metadata
        
    def _detect_geometry(self, img_to_poly: np.ndarray, doc_metadata: Tuple) -> List[Dict[str, Any]]:
        """Detecta geometría usando solo el detector de PaddleOCR."""
        
        if not self.engine:
            logger.error("El motor de geometría de PaddleOCR no está inicializado.")
            return []
        
        logger.info(f"-> Ejecutando detección geométrica en imagen de {img_to_poly.shape}...")
        
        try:
            results = self.engine.ocr(img_to_poly, cls=False, rec=False)
            
            if results and len(results) > 0 and results[0] is not None:
                logger.info(f"-> Detección geométrica exitosa. Polígonos crudos encontrados: {len(results[0])}")
            else:
                logger.warning("La detección geométrica no encontró polígonos de texto.")
                return []
                
        except Exception as e:
            logger.error(f"Error durante la detección geométrica con PaddleOCR: {e}", exc_info=True)
            return []

        polygons = []
        for idx, bbox_polygon_raw in enumerate(results[0]):
            if not isinstance(bbox_polygon_raw, list) or len(bbox_polygon_raw) < 3:
                continue
            
            try:
                # Metada del documento presentada como 'doc_metadata'
                if isinstance(doc_metadata, tuple) and len(doc_metadata) == 2:
                    dimensiones, dpi_img = doc_metadata
                    if isinstance(dimensiones, tuple) and len(dimensiones) == 2:
                        final_h, final_w = dimensiones

                    else:
                        final_h = 0
                        final_w = 0
                else:
                    final_h = 0
                    final_w = 0
                    dpi_img = 0
                    
                final_h = int(final_h)
                final_w = int(final_w)
                dpi_img = int(dpi_img)
                # Asegurar que las coordenadas son flotantes
                polygon_coords = [[float(p[0]), float(p[1])] for p in bbox_polygon_raw]
                xs = [pt[0] for pt in polygon_coords]
                ys = [pt[1] for pt in polygon_coords]
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                bbox = [xmin, ymin, xmax, ymax]
                cx = float(sum(xs)) / len(xs)
                cy = float(sum(ys)) / len(ys)
                # Cálculo de área usando la fórmula del polígono simple
                area = 0.5 * abs(sum(xs[i] * ys[(i+1)%len(xs)] - xs[(i+1)%len(xs)] * ys[i] for i in range(len(xs))))
                
                poly_dict = {
                    'polygon_id': f'poly_{idx:04d}',
                    'geometry': {
                        'polygon_coords': polygon_coords,
                        'bounding_box': bbox,
                        'centroid': [cx, cy],
                        'area': area
                    },
                    'doc_metadata': {
                        'doc_dimensions': {
                            'height': final_h,
                            'width': final_w
                        },
                        'dpi_img': dpi_img
                    }
                }
                
                polygons.append(poly_dict)
                
            except (TypeError, ValueError, IndexError):
                continue
                
        logger.info(f"-> Polígonos válidos procesados: {len(polygons)}")
        
        return polygons

    def _get_polygons(self, clean_img: np.ndarray, dpi_img: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Retorna la imagen deskewed y una lista de diccionarios de polígonos."""
        # Primero aplicar deskew usando OpenCV
        deskewed_img, doc_metadata = self._detect_angle(clean_img, dpi_img)
        
        # Copiamos para no modificar la imagen original que podría ser usada en otro lugar
        img_for_geometry = deskewed_img.copy()

        # Detectar geometría para obtener las coordenadas de los polígonos
        polygons = self._detect_geometry(img_for_geometry, doc_metadata)
        logger.debug(f"Detectados {len(polygons)} polígonos en la fase de geometría.")
                
        # Retornar la imagen corregida y la lista de polígonos
        return deskewed_img, polygons
        