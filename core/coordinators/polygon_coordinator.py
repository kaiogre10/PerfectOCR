# PerfectOCR/core/coordinators/geometric_coordinator.py
import cv2
import numpy as np
import logging
import time
import os
from typing import Any, Optional, Dict, Tuple, List
from core.workflow.geometry.cleanner import ImageCleaner
from core.workflow.geometry.deskew import Deskewer
from core.workflow.geometry.lineal_reconstructor import LineReconstructor
from core.workflow.geometry.poly_gone import PolygonExtractor
from core.workflow.geometry.binarization import Binarizator
from core.workspace.utils.output_handlers import ImageOutputHandler

logger = logging.getLogger(__name__)

class PolygonCoordinator:
    """
    Coordina la fase de preprocesamiento, delegando todo el trabajo
    a un único worker autosuficiente.
    """
    def __init__(self, config: Dict, project_root: str):
        self.project_root = project_root
        self.workflow_config = config.get('workflow', {})
        self.output_config = config.get('output_config', {})
        quality_rules = config.get('polygonal', {}).get('polygon_config', {})
        
        # Instanciar workers con sus configuraciones específicas
        self._clean = ImageCleaner(config=quality_rules.get('basic', {}), project_root=self.project_root)
        self._deskewer = Deskewer(config=quality_rules.get('deskew', {}), project_root=self.project_root)
        self._lineal = LineReconstructor(config=quality_rules.get('basic', {}), project_root=self.project_root)
        self._poly = PolygonExtractor(config=quality_rules.get('basic', {}), project_root=self.project_root)
        self._bin = Binarizator(config=quality_rules.get('binarize', {}), project_root=self.project_root)

        self.image_saver = ImageOutputHandler()
        
    def _generate_polygons(
        self,
        image_array: np.ndarray,
        input_path: str = ""
    ) -> Tuple[Optional[Dict[str, Any]], float]:
    
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY) if len(image_array.shape) > 2 else image_array
        pipeline_start = time.time()
        logger.info("=== INICIANDO GENERACIÓN DE POLIGONAL ===")
        
        # Limpieza rápida
        step_start = time.time()
        logger.info("Iniciando limpieza rapida")
        clean_img, dpi_img = self._clean._quick_enhance(gray_image, input_path)
        step_duration = time.time() - step_start
        logger.info(f"Limpieza completada en {step_duration:.4f}s")

        # Detección geométrica
        step_start = time.time()
        logger.info("Iniciando corrección de inclinación")
        deskewed_img, polygons, metadata = self._deskewer._get_polygons(clean_img, dpi_img)
        step_duration = time.time() - step_start
        logger.info(f"Corrección de inclinación completada en {step_duration:.4f}s")
        
        # Agrupamiento de líneas
        step_start = time.time()
        logger.info("Agrupando líneas")
        reconstructed_lines, metadata = self._lineal._reconstruct_lines(polygons, metadata)
        step_duration = time.time() - step_start
        logger.info(f"Agrupamiento completado en {step_duration:.4f}s - {len(reconstructed_lines)} líneas detectadas")
        
        # Añadir logs detallados por línea
        for line in reconstructed_lines:
            logger.debug(f"Línea {line['line_id']}: {len(line['polygons'])} polígonos, bbox={line['line_bbox']}")
        
        # Recorte de líneas y añadir imágenes recortadas
        step_start = time.time()
        logger.info("Iniciando recorte de líneas de imagen")
        result, individual_polygons = self._poly._add_cropped_images_to_lines(deskewed_img, reconstructed_lines, metadata, input_path, self.output_config)
        step_duration = time.time() - step_start
        logger.info(f"Recorte completado en {step_duration:.4f}s")
        
        polygons_to_binarize = individual_polygons.copy()
        # 8. Binarización por separado (solo para las features)
        step_start = time.time()
        logger.info("Iniciando binarización...")
        binarized_poly = self._bin._process_individual_polygons(polygons_to_binarize)
        step_duration = time.time() - step_start
        logger.info(f"Binarización completada en {step_duration:.4f}s")

        
        # Verificar que se añadieron las imágenes recortadas
        if result and result.get("lines"):
            lines_with_images = sum(1 for line in result["lines"] if line.get("cropped_img") is not None)
            logger.info(f"Imágenes recortadas exitosamente: {lines_with_images}/{len(result['lines'])}")

        pipeline_end = time.time() - pipeline_start
        logger.info(f"=== GENERACIÓN DE POLIGONAL COMPLETADA en {pipeline_end:.4f}s ===")
        
        return result, pipeline_end