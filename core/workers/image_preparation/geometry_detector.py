# PerfectOCR/core/workers/polygonal/geometry_detector.py
import os
import logging
from typing import Dict, Any
import numpy as np
import time
from paddleocr import PaddleOCR
from core.domain.workflow_job import Polygon, BoundingBox, PolygonGeometry
from core.workers.factory.abstract_worker import AbstractWorker

logger = logging.getLogger(__name__)

class GeometryDetector(AbstractWorker):
    """
    Worker especializado que utiliza PaddleOCR para detectar la geometría del texto.
    Implementa AbstractWorker para ser usado en la factory.
    """
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.paddle_config = config
        self._engine = None 
        logger.info("GeometryDetector inicializado (motor PaddleOCR no cargado aún).")

    @property
    def engine(self):
        """Inicialización perezosa de PaddleOCR para no consumir recursos si no se usa."""
        if self._engine is None:
            start_time = time.perf_counter()
            try:
                init_params = {
                    "use_angle_cls": False,
                    "rec": False,
                    "lang": self.paddle_config.get('lang', 'es'),
                    "show_log": self.paddle_config.get('show_log', False),
                    "use_gpu": self.paddle_config.get('use_gpu', False),
                    "enable_mkldnn": self.paddle_config.get('enable_mkldnn', True)
                }

                det_model_path = self.paddle_config.get('det_model_dir', "")
                if det_model_path and os.path.exists(det_model_path):
                    init_params['det_model_dir'] = det_model_path
                else:
                    logger.warning("No se encontró un directorio de modelo de detección local. PaddleOCR intentará descargarlo.")

                model_load_start = time.perf_counter()
                self._engine = PaddleOCR(**init_params)
                model_load_time = time.perf_counter() - model_load_start
                total_init_time = time.perf_counter() - start_time
                logger.info(f"Total inicialización PaddleOCR para GEOMETRÍA: {total_init_time:.3f}s (carga de modelo: {model_load_time:.3f}s)")

                logger.info("Instancia de PaddleOCR para GEOMETRÍA cargada exitosamente.")

            except Exception as e:
                logger.error(f"Error crítico al inicializar la instancia geométrica de PaddleOCR: {e}", exc_info=True)
                self._engine = None
        return self._engine

    def process(self, image: np.ndarray[Any, Any], context: Dict[str, Any]) -> np.ndarray[Any, Any]:
        """
        Implementa el contrato de AbstractWorker.
        Detecta geometría y actualiza el WorkflowJob en el contexto.
        """
        workflow_job = context.get('workflow_job')
        if not workflow_job:
            logger.warning("GeometryDetector: No se encontró WorkflowJob en el contexto")
            return image
        
        try:
            # Usar el método detect existente
            doc_data = {
                "metadata": workflow_job.doc_metadata.__dict__ if workflow_job.doc_metadata else {},
                "polygons": {}
            }
            
            doc_data = self.detect(image, doc_data)
            
            # Actualizar WorkflowJob con los polígonos detectados
            if doc_data["polygons"]:
                logger.info(f"[GeometryDetector] Encontrados {len(doc_data['polygons'])} polígonos")
                
                # Convertir polígonos detectados a objetos Polygon del WorkflowJob
                for poly_id, poly_data in doc_data['polygons'].items():
                    try:
                        geometry_data = poly_data['geometry']
                
                        bbox = geometry_data['bounding_box']
                        bounding_box = BoundingBox(
                            x_min=bbox[0], y_min=bbox[1], 
                            x_max=bbox[2], y_max=bbox[3]
                        )
                        
                        polygon_geometry = PolygonGeometry(
                            polygon_coords=geometry_data['polygon_coords'],
                            bounding_box=bounding_box,
                            centroid=tuple(geometry_data['centroid']),
                            perimeter=None 
                        )
                        
                        polygon = Polygon(
                            polygon_id=poly_id,
                            geometry=polygon_geometry,
                            line_id=None,
                            cropped_img=None,
                            padding_coords=None,
                            was_fragmented=False,
                            text=None,
                            confidence=None
                        )
                        
                        workflow_job.add_polygon(polygon)
                        
                    except Exception as e:
                        logger.warning(f"Error convirtiendo polígono {poly_id}: {e}")
                        continue
                
                logger.info(f"[GeometryDetector] {len(doc_data['polygons'])} polígonos guardados en WorkflowJob")
            
            else:
                logger.warning("[GeometryDetector] No se encontraron polígonos")
            
        except Exception as e:
            error_msg = f"Error en GeometryDetector: {e}"
            logger.error(error_msg, exc_info=True)
            if workflow_job:
                workflow_job.add_error(error_msg)
        
        return image  # Retorna la misma imagen (no la modifica)

    def detect(self, current_image: np.ndarray[Any, Any], doc_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detecta la geometría del texto y la añade al diccionario de datos del documento.
        """
        if not self.engine:
            logger.error("El motor de geometría de PaddleOCR no está inicializado.")
            doc_data["polygons"] = {}
            return doc_data
        
        try:
            results = self.engine.ocr(current_image, cls=False, rec=False)
            if not (results and len(results) > 0 and results[0] is not None):
                logger.warning("La detección geométrica no encontró polígonos de texto.")
                doc_data['polygons'] = {}
                return doc_data
        except Exception as e:
            logger.error(f"Error durante la detección geométrica con PaddleOCR: {e}", exc_info=True)
            doc_data['polygons'] = {}
            return doc_data

        polygons_dict = {}
        for idx, bbox_polygon_raw in enumerate(results[0]):
            if not isinstance(bbox_polygon_raw, list) or len(bbox_polygon_raw) < 3:
                continue
            
            try:
                polygon_coords = [[float(p[0]), float(p[1])] for p in bbox_polygon_raw]
                xs = [pt[0] for pt in polygon_coords]
                ys = [pt[1] for pt in polygon_coords]
                
                poly_id = f'poly_{idx:04d}'
                poly_entry = {
                    'polygon_id': poly_id,
                    'geometry': {
                        'polygon_coords': polygon_coords,
                        'bounding_box': [min(xs), min(ys), max(xs), max(ys)],
                        'centroid': [float(sum(xs)) / len(xs), float(sum(ys)) / len(ys)],
                    }
                }
                polygons_dict[poly_id] = poly_entry

            except (TypeError, ValueError, IndexError) as e:
                logger.warning(f"Omitiendo polígono inválido en el índice {idx}: {e}")
                continue
                
        doc_data["polygons"] = polygons_dict
        
        return doc_data
