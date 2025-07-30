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
from core.workflow.geometry.fragmentator import PolygonFragmentator
from core.workspace.utils.output_handlers import ImageOutputHandler

logger = logging.getLogger(__name__)

class PolygonCoordinator:
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
        self.image_saver = ImageOutputHandler()
                
    def _generate_polygons(
        self,
        image_array: np.ndarray,
        input_path: str
    ) -> Tuple[Optional[List[Dict[str, Any]]], float]:
    
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
        deskewed_img, polygons = self._deskewer._get_polygons(clean_img, dpi_img)
        step_duration = time.time() - step_start
        logger.info(f"Corrección de inclinación completada en {step_duration:.4f}s")
            
        # Agrupamiento de líneas
        step_start = time.time()
        logger.info("Agrupando líneas")
        # Esta función ahora enriquece los polígonos existentes con 'line_id'
        lineal_polygons = self._lineal._reconstruct_lines(polygons)
        step_duration = time.time() - step_start
        logger.info(f"Agrupamiento completado en {step_duration:.4f}s - {len(polygons)} polígonos detectados")
                
        # Procesar polígonos individuales directamente
        step_start = time.time()
        logger.info("Iniciando recorte de polígonos de imagen")
        # Esta función añade 'cropped_img' a cada polígono
        extracted_polygons = self._poly._extract_individual_polygons(deskewed_img, lineal_polygons)
        step_duration = time.time() - step_start
        logger.info(f"Recorte completado en {step_duration:.6f}s")
        
        if not polygons:
            logger.warning("No se encontraron polígonos para procesar")
            return None, time.time() - pipeline_start
        
        poly_with_images = sum(1 for poly in extracted_polygons if poly.get("cropped_img") is not None)
        if poly_with_images is None:
            
            return poly_with_images
        
        # Binarización de polígonos individuales
        step_start = time.time()
        logger.info("Iniciando binarización...")
        # El binarizador devuelve una copia binarizada para análisis y la lista original intacta
        binarized_poly, individual_polygons = self._bin._binarize_polygons(extracted_polygons)
        step_duration = time.time() - step_start
        logger.info(f"Binarización completada en {step_duration:.4f}s")
        
        if not individual_polygons:
            logger.warning("No se devolvieron polígonos originales tras la binarización")
            return None, time.time() - pipeline_start
        
        # Verificar binarización
        binarized_count = sum(1 for poly in binarized_poly if poly.get("binarized_img") is not None)
        if binarized_count is None:
            return binarized_count
        
        # Preparación final de polígonos (Fragmentación)
        step_start = time.time()
        # El fragmentador usa los binarizados para medir y los originales para actuar
        refined_polygons = self._fragment._intercept_polygons(binarized_poly, individual_polygons)
        step_duration = time.time() - step_start
        
        if input_path:
            base_name = os.path.basename(input_path)
            file_name, _ = os.path.splitext(base_name)
            output_folder = self.workflow_config.get('output_folder')
            output_flags = self.output_config.get('enabled_outputs', {})
                        
            # Guardar refined_polygons si está habilitado
            if output_flags.get('refined_polygons') and output_folder:
                self._save_polygon_images(refined_polygons, output_folder, f"{file_name}_refined_polygons")
            
            # Guardar problematic_ids si está habilitado  
            if output_flags.get('problematic_ids', False) and output_folder:
                problematic_ids = self._fragment._get_problematic_ids()
                logger.info(f"IDs problemáticos encontrados: {len(problematic_ids) if problematic_ids else 0}")
                if problematic_ids:
                    problematic_polygons = [poly for poly in refined_polygons if poly.get('polygon_id') in problematic_ids]
                    logger.info(f"Intentando guardar {len(problematic_polygons)} problematic_polygons en '{output_folder}'")
                    self._save_polygon_images(problematic_polygons, output_folder, f"{file_name}_problematic_ids")
                else:
                    logger.warning("No se encontraron IDs problemáticos para guardar")
            else:
                logger.warning(f"No se guardaron problematic_ids - Habilitado: {output_flags.get('problematic_ids')}, Carpeta válida: {bool(output_folder)}")

        pipeline_end = time.time() - pipeline_start
        logger.info(f"=== GENERACIÓN DE POLIGONAL COMPLETADA en {pipeline_end:.4f}s ===")
        
        return refined_polygons, pipeline_end

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