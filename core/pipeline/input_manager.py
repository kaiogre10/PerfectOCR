# PerfectOCR/core/coordinators/geometric_coordinator.py
import cv2
import numpy as np
import logging
import time
import os
from typing import Any, Optional, Dict, Tuple, List, Set
from concurrent.futures import ThreadPoolExecutor
from core.workers.image_preparation.image_loader import ImageLoader
from core.workers.image_preparation.cleanner import ImageCleaner
from core.workers.image_preparation.angle_corrector import AngleCorrector
from core.workers.image_preparation.geometry_detector import GeometryDetector
from core.workers.image_preparation.lineal_reconstructor import LineReconstructor
from core.workers.image_preparation.poly_gone import PolygonExtractor

logger = logging.getLogger(__name__)

class InputManager:
    """
    Coordina la fase de extracción de polígonos, delegando todo el trabajo
    a un único worker autosuficiente.
    """
    def __init__(self, config: Dict, stage_config: Dict, project_root: str):
        self.project_root = project_root
        self.image_loader = config
        paddle_config = self.image_loader.get('paddle_config', {})
        self._image_load = ImageLoader(config=self.image_loader.get('extensions', {}), project_root=self.project_root)
        self._cleaner = ImageCleaner(config=self.image_loader.get('cleaning', {}), project_root=self.project_root)
        self._angle_corrector = AngleCorrector(config=self.image_loader.get('deskew', {}), project_root=self.project_root)
        self._geometry_detector = GeometryDetector(paddle_config=paddle_config, project_root=self.project_root)
        self._poly = PolygonExtractor(config=self.image_loader.get('cutting', {}), project_root=self.project_root)
        self._lineal = LineReconstructor(config={}, project_root=self.project_root)
        self._lines_geometry: Optional[Dict[str, Any]] = None
        self._binarization: Optional[Dict[str, Any]] = None
                
    def _generate_polygons(
        self, config: Dict[str, Any],
        output_flags: Dict[str, bool]
    ) -> Tuple[Optional[Dict[str, Any]], float]:
        
        pipeline_start = time.time()
        logger.info("INICIANDO GENERACIÓN DE POLIGONAL")

        gray_image, metadata = self._image_load._load_image_and_metadata()
        
        clean_image, doc_data = self._cleaner._quick_enhance(gray_image)
        
        step_start = time.time()
        logger.info("Iniciando fase de corrección de ángulo y detección de geometría.")
        
        metadata = doc_data.get("metadata", {})
        img_dims = metadata.get("img_dims", {})
        h = img_dims.get("height")
        w = img_dims.get("width")

        if not h or not w:
            logger.warning("Dimensiones no encontradas en metadata, extrayendo de la imagen.")
            h, w = clean_image.shape[:2]
            img_dims = {"height": h, "width": w}
            metadata["img_dims"] = img_dims
        
        # 1. Corregir ángulo
        logger.info("Iniciando corrección de inclinación.")
        deskewed_img, new_dims = self._angle_corrector.correct(clean_image, img_dims)
        
        # Actualizar el diccionario de dimensiones original en su lugar.
        img_dims.update(new_dims)
        img_for_geometry = deskewed_img.copy()

        # 2. Detectar geometría
        logger.info("Iniciando detección de geometría.")
        enriched_doc = self._geometry_detector.detect(img_for_geometry, doc_data)

        step_duration = time.time() - step_start
        polygons_count = len(enriched_doc.get("polygons", {}))
        logger.info(f"Fase de geometría completada en {step_duration:.4f}s. Detectados {polygons_count} polígonos.")
            
        # Paralelización de lineal y poly
        step_start_parallel = time.time()
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_lineal = executor.submit(self._lineal._reconstruct_lines, enriched_doc)
            future_poly = executor.submit(self._poly._extract_individual_polygons, deskewed_img, enriched_doc)
            lineal_result = future_lineal.result()
            extracted_polygons = future_poly.result()
               
        total_lines = self._lineal._get_lines_geometry()
                
        step_duration_parallel = time.time() - step_start_parallel
        logger.info(f"Paralelización lineal/poly completada en {step_duration_parallel:.4f}s")
        logger.info(f"Lineal devolvió {len(lineal_result)} IDs y {len(total_lines)} geometrías de línea. Poly devolvió {len(extracted_polygons)} polígonos.")

        fusionados = 0
        for poly_id, poly in extracted_polygons.items():
            pid = poly.get("polygon_id")
            if pid and pid in lineal_result:
                poly["line_id"] = lineal_result[pid]
                fusionados += 1
        logger.info(f"Fusión de line_id completada: {fusionados} polígonos enriquecidos.")
        
        time_poly = time.time () - pipeline_start
        
        return extracted_polygons, time_poly

    def _get_polygons_to_binarize(self) -> Dict[str, Any]:
        """Expone los IDs problemáticos para que el builder pueda usarlos."""
         
        polygons_to_binarize = self._poly._get_polygons_copy()
        
        polygons_to_bin = polygons_to_binarize
        
        return polygons_to_bin
