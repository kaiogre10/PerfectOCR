# PerfectOCR/core/coordinators/geometric_coordinator.py
import cv2
import numpy as np
import logging
import time
import os
from typing import Any, Optional, Dict, Tuple, List
from core.workflow.preprocessing.cleanner import ImageCleaner
from core.workflow.geometry.deskew import Deskewer
from core.workflow.geometry.lineal_reconstructor import LineReconstructor
from core.workflow.geometry.poly_gone import PolygonExtractor
from core.workflow.geometry.multifeaturer import MultiFeacturer
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
        quality_rules = config.get('polygon_config', {})
        
        # Instanciar workers con sus configuraciones específicas
        self._clean = ImageCleaner(config=quality_rules.get('basic', {}), project_root=self.project_root)
        self._deskewer = Deskewer(config=quality_rules.get('deskew', {}), project_root=self.project_root)
        self._lineal = LineReconstructor(config=quality_rules.get('basic', {}), project_root=self.project_root)
        self._poly = PolygonExtractor(config=quality_rules.get('basic', {}), project_root=self.project_root)
        self._features = MultiFeacturer()

        self.image_saver = ImageOutputHandler()
        
    def _generate_polygons(
        self,
        image_array: np.ndarray,
        input_path: str = ""
    ) -> Tuple[Optional[Dict[str, Any]], float]:
    
        try:
            gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY) if len(image_array.shape) > 2 else image_array
            pipeline_start = time.time()
            logger.info("=== INICIANDO GENERACIÓN DE POLIGONAL ===")
            
            # Limpieza rápida
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
            step_star = time.time()
            logger.info("Agrupando líneas")
            reconstructed_lines = self._lineal._reconstruct_lines(polygons, metadata)
            step_duration = time.time() - step_start
            logger.info(f"Agrupamiento completado en {step_duration:.4f}s")