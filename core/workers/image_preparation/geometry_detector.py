# PerfectOCR/core/workers/polygonal/geometry_detector.py
import os
import logging
from typing import Dict, Any
import numpy as np
from paddleocr import PaddleOCR
from core.workers.factory.abstract_worker import AbstractWorker
from core.domain.workflow_job import ProcessingStage

logger = logging.getLogger(__name__)

class GeometryDetector(AbstractWorker):
    """
    Worker especializado que utiliza PaddleOCR para detectar la geometría del texto.
    """
    def __init__(self, paddle_config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.paddle_config = paddle_config
        self._engine = None 
        logger.info("GeometryDetector inicializado (motor PaddleOCR no cargado aún).")

    def process(self, image: np.ndarray[Any, Any], context: Dict[str, Any]) -> np.ndarray[Any, Any]:
        """
        Implementa el método abstracto de AbstractWorker.
        """
        workflow_job = context.get('workflow_job')
        metadata = context.get('metadata', {})
        
        # Crear doc_data para compatibilidad
        doc_data = {
            "metadata": metadata,
            "polygons": {}
        }
        
        # Detectar geometría
        doc_data = self.detect(image, doc_data)
        
        # Actualizar el WorkflowJob si está disponible
        if workflow_job and workflow_job.full_img is not None:
            workflow_job.update_stage(ProcessingStage.GEOMETRY_DETECTED)
            # Aquí podrías agregar los polígonos al job si es necesario
        
        return image  # Retorna la misma imagen (no la modifica)

    @property
    def engine(self):
        """Inicialización perezosa de PaddleOCR para no consumir recursos si no se usa."""
        if self._engine is None:
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

                self._engine = PaddleOCR(**init_params)
                logger.info("Instancia de PaddleOCR para GEOMETRÍA cargada exitosamente.")

            except Exception as e:
                logger.error(f"Error crítico al inicializar la instancia geométrica de PaddleOCR: {e}", exc_info=True)
                self._engine = None
        return self._engine

    def detect(self, img_to_poly: np.ndarray[Any, Any], doc_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detecta la geometría del texto y la añade al diccionario de datos del documento.
        """
        if not self.engine:
            logger.error("El motor de geometría de PaddleOCR no está inicializado.")
            doc_data["polygons"] = {}
            return doc_data
        
        try:
            results = self.engine.ocr(img_to_poly, cls=False, rec=False)
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
