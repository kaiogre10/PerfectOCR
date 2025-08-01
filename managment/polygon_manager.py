# PerfectOCR/core/coordinators/geometric_coordinator.py
import cv2
import numpy as np
import logging
import time
import os
from typing import Any, Optional, Dict, Tuple, List
from concurrent.futures import ThreadPoolExecutor
from core.polygonal.cleanner import ImageCleaner
from core.polygonal.deskew import Deskewer
from core.polygonal.lineal_reconstructor import LineReconstructor
from core.polygonal.poly_gone import PolygonExtractor
from core.polygonal.binarization import Binarizator
from core.polygonal.fragmentator import PolygonFragmentator
#from core.utils.output_handlers import ImageOutputHandler

logger = logging.getLogger(__name__)

class PolygonManager:
    """
    Coordina la fase de extracción de polígonos, delegando todo el trabajo
    a un único worker autosuficiente.
    """
    def __init__(self, config: Dict, project_root: str):
        self.project_root = project_root
        self.workflow_config = config.get('workflow', {})
        self.output_config = config.get('output_config', {})
        polygonal_params = config.get('polygonal', {})

        self._clean = ImageCleaner(config=polygonal_params.get('polygon_config', {}), project_root=self.project_root)
        self._deskewer = Deskewer(config=polygonal_params.get('polygon_config', {}), project_root=self.project_root)
        self._lineal = LineReconstructor(config=polygonal_params.get('polygon_config', {}), project_root=self.project_root)
        self._poly = PolygonExtractor(config=polygonal_params.get('polygon_config', {}), project_root=self.project_root)
        self._bin = Binarizator(config=polygonal_params.get('polygon_config', {}), project_root=self.project_root)
        self._fragment = PolygonFragmentator(config=polygonal_params.get('polygon_config', {}), project_root=self.project_root)
        #self.image_saver = ImageOutputHandler()
        self._lines_geometry: Optional[Dict[str, Any]] = None
        self._binarization: Optional[Dict[str, Any]] = None
                
    def _generate_polygons(
        self,
        image_array: np.ndarray,
        input_path: str
    ) -> Tuple[Optional[Dict[str, Any]], float]:
    
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY) if len(image_array.shape) > 2 else image_array
        pipeline_start = time.time()
        logger.info("INICIANDO GENERACIÓN DE POLIGONAL")
        
        # Limpieza rápida
        step_start = time.time()
        logger.info("Iniciando limpieza rapida")
        clean_img, doc_data = self._clean._quick_enhance(gray_image, input_path)
        step_duration = time.time() - step_start
        logger.info(f"Limpieza completada en {step_duration:.4f}s")

        # Detección geométrica
        step_start = time.time()
        logger.info("Iniciando corrección de inclinación")
        deskewed_img, enriched_doc = self._deskewer._get_polygons(clean_img, doc_data)
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
        for poly_id, poly in extracted_polygons.items(): # <--- CAMBIO 1
            # pid ahora es poly_id directamente, pero mantenemos la lógica por seguridad
            pid = poly.get("polygon_id")
            if pid and pid in lineal_result:
                poly["line_id"] = lineal_result[pid]
                fusionados += 1
        logger.info(f"Fusión de line_id completada: {fusionados} polígonos enriquecidos.")

        # Binarización de polígonos individuales
        step_start = time.time()
        logger.info("Iniciando binarización")
        binarized_polygons = self._bin._binarize_polygons(polygons_to_bin)
        step_duration = time.time() - step_start
        logger.info(f"Binarización completada en {step_duration:.4f}s")
        
        # Verificar binarización
        if binarized_polygons is None:
            logger.warning("La binarización no devolvió resultados.")
            return None, time.time() - pipeline_start
        
        binarized_count = len(binarized_polygons)
        logger.info(f"Se binarizaron {binarized_count} polígonos.")
        
        # Preparación final de polígonos (Fragmentación)
        step_start = time.time()
        # El fragmentador usa los binarizados para medir y los originales para actuar
        refined_polygons = self._fragment._intercept_polygons(binarized_polygons, extracted_polygons)
        step_duration = time.time() - step_start
        
        if input_path:
            base_name = os.path.basename(input_path)
            file_name, _ = os.path.splitext(base_name)
            output_folder = self.workflow_config.get('output_folder')
            output_flags = self.output_config.get('enabled_outputs', {})
                        
            # Guardar refined_polygons si está habilitado
            if output_flags.get('refined_polygons') and output_folder:
                self._save_polygon_images(list(refined_polygons.values()), output_folder, f"{file_name}_refined_polygons")
            
            # Guardar problematic_ids si está habilitado  
            if output_flags.get('problematic_ids', False) and output_folder:
                problematic_ids = self._fragment._get_problematic_ids()
                logger.info(f"IDs problemáticos encontrados: {len(problematic_ids) if problematic_ids else 0}")
                if problematic_ids:
                    problematic_polygons = {pid: poly for pid, poly in refined_polygons.items() if pid in problematic_ids}
                    logger.info(f"Intentando guardar {len(problematic_polygons)} problematic_polygons en '{output_folder}'")
                    self._save_polygon_images(list(problematic_polygons.values()), output_folder, f"{file_name}_problematic_ids")
                else:
                    logger.warning("No se encontraron IDs problemáticos para guardar")
                    
        pipeline_end = time.time() - pipeline_start
        logger.info(f"GENERACIÓN DE POLIGONAL COMPLETADA en {pipeline_end:.4f}s")
        
        time_poly = pipeline_end
        
        return refined_polygons, time_poly

    def _save_polygon_images(self, polygons: List[Dict], output_folder: str, base_filename: str):
        """Guarda solo las imágenes de los polígonos, no metadata."""
        if not polygons:
            return
            
        try:
            os.makedirs(output_folder, exist_ok=True)
            for i, poly in enumerate(polygons):
                img = poly.get('cropped_img')
                if img is not None:
                    img_filename = f"{base_filename}_{i+1}.png"
                    img_path = os.path.join(output_folder, img_filename)
                    cv2.imwrite(img_path, img)
            logger.info(f"Guardadas {len(polygons)} imágenes en {output_folder}")
        except Exception as e:
            logger.error(f"Error guardando imágenes de polígonos: {e}")