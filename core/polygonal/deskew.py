# PerfectOCR/core/workflow/geometry/deskew.py
import os
import cv2
import numpy as np
import logging
import math
from typing import Dict, Any, Tuple

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
        
    def _detect_angle(self, clean_img: np.ndarray, img_dims: dict) -> Tuple[np.ndarray, dict]:
        """
        Aplica deskew a la imagen si es necesario y retorna la imagen (corregida o no)
        junto con un diccionario de dimensiones siempre válido.
        """
        deskew_corrections = self.corrections
        canny = deskew_corrections.get('canny_thresholds', [50, 150])
        hough_thresh = deskew_corrections.get('hough_threshold', 150)
        max_gap = deskew_corrections.get('hough_max_line_gap_px', 20)
        angle_range = deskew_corrections.get('hough_angle_filter_range_degrees', [-15.0, 15.0])
        min_len_cap = deskew_corrections.get('hough_min_line_length_cap_px', 300)
        min_angle = deskew_corrections.get('min_angle_for_correction', 0.1)

        h = int(img_dims.get("height", 0) or 0)
        w = int(img_dims.get("width", 0) or 0)
        center = (w // 2, h // 2)
        min_len = min(clean_img.shape[1] // 3, min_len_cap)
        edges = cv2.Canny(clean_img, canny[0], canny[1])
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=hough_thresh,
                                minLineLength=min_len, maxLineGap=max_gap)

        if lines is None or len(lines) == 0:
            return clean_img, {"height": h, "width": w}

        angles = [math.degrees(math.atan2(l[0][3]-l[0][1], l[0][2]-l[0][0])) for l in lines]
        filtered = [a for a in angles if angle_range[0] < a < angle_range[1]]
        angle = np.median(filtered) if filtered else 0.0

        if abs(angle) > min_angle:
            logger.info(f"-> Aplicando corrección de inclinación: {angle:.2f} grados.")
            rotation_matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
            deskewed_img = cv2.warpAffine(clean_img, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            final_h, final_w = deskewed_img.shape[:2]
            return deskewed_img, {"height": final_h, "width": final_w}
        else:
            return clean_img, {"height": h, "width": w}
        
    def _detect_geometry(self, img_to_poly: np.ndarray, doc_data: Dict) -> Dict[str, Any]:
        """
        Detecta la geometría del texto y añade una lista de polígonos
        al diccionario de datos del documento original.
        """
        if not self.engine:
            logger.error("El motor de geometría de PaddleOCR no está inicializado.")
            return doc_data
        
        try:
            results = self.engine.ocr(img_to_poly, cls=False, rec=False)
            if not (results and len(results) > 0 and results[0] is not None):
                logger.warning("La detección geométrica no encontró polígonos de texto.")
                doc_data['polygons'] = []
                return doc_data
            logger.info(f"-> Detección geométrica exitosa. Polígonos crudos encontrados: {len(results[0])}")
        except Exception as e:
            logger.error(f"Error durante la detección geométrica con PaddleOCR: {e}", exc_info=True)
            doc_data['polygons'] = []
            return doc_data

        polygons_list = []
        for idx, bbox_polygon_raw in enumerate(results[0]):
            if not isinstance(bbox_polygon_raw, list) or len(bbox_polygon_raw) < 3:
                continue
            
            try:
                polygon_coords = [[float(p[0]), float(p[1])] for p in bbox_polygon_raw]
                xs = [pt[0] for pt in polygon_coords]
                ys = [pt[1] for pt in polygon_coords]
                
                poly_entry = {
                    'polygon_id': f'poly_{idx:04d}',
                    'geometry': {
                        'polygon_coords': polygon_coords,
                        'bounding_box': [min(xs), min(ys), max(xs), max(ys)],
                        'centroid': [float(sum(xs)) / len(xs), float(sum(ys)) / len(ys)],
                    }
                }
                polygons_list.append(poly_entry)

            except (TypeError, ValueError, IndexError) as e:
                logger.warning(f"Omitiendo polígono inválido en el índice {idx}: {e}")
                continue
                
        logger.info(f"-> Polígonos válidos procesados: {len(polygons_list)}")
        
        # Añadir la lista de polígonos al diccionario original (modificándolo en el lugar).
        doc_data["polygons"] = polygons_list
        
        return doc_data

    def _get_polygons(self, clean_img: np.ndarray, doc_data: Dict) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Retorna la imagen deskewed y el diccionario de datos enriquecido con polígonos."""
        metadata = doc_data.get("metadata", {})
        img_dims = metadata.get("img_dims", {})
        
        h = img_dims.get("height")
        w = img_dims.get("width")

        if not h or not w:
            h, w = clean_img.shape[:2]
            img_dims = {"height": h, "width": w}
            metadata["img_dims"] = img_dims
                    
        deskewed_img, new_dims = self._detect_angle(clean_img, img_dims)
        
        # Actualizar el diccionario de dimensiones original en su lugar.
        img_dims.update(new_dims)
        img_for_geometry = deskewed_img.copy()

        # Pasar el diccionario original, que ahora tiene las dimensiones actualizadas.
        # _detect_geometry lo modificará añadiendo los polígonos.
        enriched_doc = self._detect_geometry(img_for_geometry, doc_data)
        
        polygons_count = len(enriched_doc.get("polygons", []))
        logger.debug(f"Detectados {polygons_count} polígonos en la fase de geometría.")
                
        return deskewed_img, enriched_doc
