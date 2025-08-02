# PerfectOCR/core/coordinators/geometric_coordinator.py
import cv2
import numpy as np
import logging
import time
import os
from typing import Any, Optional, Dict, Tuple, List, Set
from concurrent.futures import ThreadPoolExecutor
from core.workers.polygonal.cleanner import ImageCleaner
from core.workers.polygonal.deskew import Deskewer
from core.workers.polygonal.lineal_reconstructor import LineReconstructor
from core.workers.polygonal.poly_gone import PolygonExtractor
from core.workers.polygonal.binarization import Binarizator
from core.workers.polygonal.fragmentator import PolygonFragmentator
#from core.utils.output_handlers import ImageOutputHandler

logger = logging.getLogger(__name__)

class PolygonManager:
    """
    Coordina la fase de extracción de polígonos, delegando todo el trabajo
    a un único worker autosuficiente.
    """
    def __init__(self, config: Dict, project_root: str):
        self.project_root = project_root
        # El manager ahora recibe directamente su sección del config.
        # Ya no necesita hacer config.get('polygonal', {}).
        self.polygonal_config = config
        
        # Inyección en cascada a los workers:
        self._deskewer = Deskewer(config=self.polygonal_config.get('deskew', {}), project_root=self.project_root)
        self._bin = Binarizator(config=self.polygonal_config.get('binarize', {}), project_root=self.project_root)
        self._fragment = PolygonFragmentator(config=self.polygonal_config.get('fragmentation', {}), project_root=self.project_root)
        self._poly = PolygonExtractor(config=self.polygonal_config.get('cutting', {}), project_root=self.project_root)
        self._lineal = LineReconstructor(config={}, project_root=self.project_root) # No tiene config específica
        
        self._lines_geometry: Optional[Dict[str, Any]] = None
        self._binarization: Optional[Dict[str, Any]] = None
                
    def _generate_polygons(
        self,
        image_array: np.ndarray,
        input_path: str
    ) -> Tuple[Optional[Dict[str, Any]], float]:
    
        # Ya no se necesita ImageCleaner, la carga la hace el builder.
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY) if len(image_array.shape) > 2 else image_array
        pipeline_start = time.time()
        logger.info("INICIANDO GENERACIÓN DE POLIGONAL")

        # Detección geométrica
        step_start = time.time()
        logger.info("Iniciando corrección de inclinación")
        # El doc_data ahora se crea aquí adentro.
        doc_data = {'metadata': {'doc_name': os.path.basename(input_path)}}
        deskewed_img, enriched_doc = self._deskewer._get_polygons(gray_image, doc_data)
        step_duration = time.time() - step_start
        logger.info(f"Corrección de inclinación completada en {step_duration:.4f}s")
            
        # Paralelización de lineal y poly
        step_start_parallel = time.time()
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_lineal = executor.submit(self._lineal._reconstruct_lines, enriched_doc)
            future_poly = executor.submit(self._poly._extract_individual_polygons, deskewed_img, enriched_doc)
            lineal_result = future_lineal.result()
            extracted_polygons = future_poly.result()
               
        self._lines_geometry = self._lineal._get_lines_geometry()
        polygons_to_bin = self._poly._get_polygons_copy()
        
        step_duration_parallel = time.time() - step_start_parallel
        logger.info(f"Paralelización lineal/poly completada en {step_duration_parallel:.4f}s")
        logger.info(f"Lineal devolvió {len(lineal_result)} IDs y {len(self._lines_geometry)} geometrías de línea. Poly devolvió {len(extracted_polygons)} polígonos.")

        fusionados = 0
        for poly_id, poly in extracted_polygons.items():
            pid = poly.get("polygon_id")
            if pid and pid in lineal_result:
                poly["line_id"] = lineal_result[pid]
                fusionados += 1
        logger.info(f"Fusión de line_id completada: {fusionados} polígonos enriquecidos.")

        step_start = time.time()
        logger.info("Iniciando binarización")
        binarized_polygons = self._bin._binarize_polygons(polygons_to_bin)
        step_duration = time.time() - step_start
        logger.info(f"Binarización completada en {step_duration:.4f}s")
        
        if binarized_polygons is None:
            logger.warning("La binarización no devolvió resultados.")
            return None, time.time() - pipeline_start
        
        binarized_count = len(binarized_polygons)
        logger.info(f"Se binarizaron {binarized_count} polígonos.")
        
        step_start = time.time()
        refined_polygons = self._fragment._intercept_polygons(binarized_polygons, extracted_polygons)
        step_duration = time.time() - step_start
        
        
        # El guardado de imágenes ahora es responsabilidad del ProcessingBuilder.
        # Se elimina el código de guardado de aquí.
                    
        pipeline_end = time.time() - pipeline_start
        logger.info(f"GENERACIÓN DE POLIGONAL COMPLETADA en {pipeline_end:.4f}s")
        
        time_poly = pipeline_end
        
        return refined_polygons, time_poly

    def get_problematic_ids(self) -> Set[str]:
        """Expone los IDs problemáticos para que el builder pueda usarlos."""
        return self._fragment._get_problematic_ids()