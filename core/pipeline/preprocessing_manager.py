# PerfectOCR/core/coordinators/preprocessing_coordinator.py
import cv2
import numpy as np
import logging
import time
import os
from typing import Any, Optional, Dict, Tuple, List, Set
from core.workers.preprocessing.moire import MoireDenoiser
from core.workers.preprocessing.sp import DoctorSaltPepper
from core.workers.preprocessing.gauss import GaussianDenoiser
from core.workers.preprocessing.clahe import ClaherEnhancer
from core.workers.preprocessing.sharp import SharpeningEnhancer
from core.workers.preprocessing.binarization import Binarizator
from core.workers.preprocessing.fragmentator import PolygonFragmentator

logger = logging.getLogger(__name__)

class PreprocessingManager:
    """
    Coordina la fase de preprocesamiento, delegando todo el trabajo a un único worker autosuficiente.
    """
    def __init__(self, config: Dict, stage_config: Dict, project_root: str):
        self.project_root = project_root
        self.preprocessing_config = config
        self.manager_config = stage_config
        denoise_config = self.preprocessing_config.get('denoise', {})
        
        self._moire = MoireDenoiser(config=denoise_config.get('moire', {}), project_root=self.project_root)
        self._sp = DoctorSaltPepper(config=denoise_config.get('median_filter', {}), project_root=self.project_root)
        self._gauss = GaussianDenoiser(config=denoise_config.get('bilateral_params', {}), project_root=self.project_root)
        self._claher = ClaherEnhancer(config=self.preprocessing_config.get('contrast', {}), project_root=self.project_root)
        self._sharp = SharpeningEnhancer(config=self.preprocessing_config.get('sharpening', {}), project_root=self.project_root)
        self._fragment = PolygonFragmentator(config=self.preprocessing_config.get('fragmentation', {}), project_root=self.project_root)
        self._bin = Binarizator(config=self.preprocessing_config.get('binarize', {}), project_root=self.project_root)
        
    def _apply_preprocessing_pipelines(self, extracted_polygons: Dict[str, Any], polygons_to_bin: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Procesa el diccionario general de forma secuencial:
        1. Moiré
        2. Ruido sal y pimienta
        3. Ruido general
        4. Contraste
        5. Nitidez
        El diccionario principal se va enriqueciendo y se devuelve el mismo (con imágenes modificadas en 'cropped_img').
        """
        pipeline_start = time.time()
        logger.info("Iniciando pipeline de preprocesamiento")

        # Verificar que hay polígonos para procesar
        polygons = extracted_polygons.get("polygons", {})
        if not polygons:
            logger.warning("No se encontraron polígonos para preprocesar")
            return extracted_polygons, 0.0
        
        logger.info(f"Procesando {len(polygons)} polígonos")

        # Etapa 1: Detección de patrones Moiré
        moire_start = time.time()
        logger.info("Iniciando detección de patrones Moiré")
        moire_dict = self._moire._detect_moire_patterns(extracted_polygons)
        moire_duration = time.time() - moire_start
        logger.info(f"Detección de patrones Moiré completada en {moire_duration:.3f} segundos")

        # Etapa 2: Estimación de ruido sal y pimienta
        sp_start = time.time()
        logger.info("Iniciando estimación de ruido sal y pimienta")
        sp_dict = self._sp._estimate_salt_pepper_noise(moire_dict)
        sp_duration = time.time() - sp_start
        logger.info(f"Estimación de ruido sal y pimienta completada en {sp_duration:.3f} segundos")

        # Etapa 3: Estimación de ruido gaussiano
        gauss_start = time.time()
        logger.info("Iniciando estimación de ruido gaussiano")
        gauss_dict = self._gauss._estimate_gaussian_noise(sp_dict)
        gauss_duration = time.time() - gauss_start
        logger.info(f"Estimación de ruido gaussiano completada en {gauss_duration:.3f} segundos")

        # Etapa 4: Estimación de contraste
        clahe_start = time.time()
        logger.info("Iniciando estimación de contraste")
        clahe_dict = self._claher._estimate_contrast(gauss_dict)
        clahe_duration = time.time() - clahe_start
        logger.info(f"Estimación de contraste completada en {clahe_duration:.3f} segundos")

        # Etapa 5: Estimación de nitidez
        sharp_start = time.time()
        logger.info("Iniciando estimación de nitidez")
        sharp_dict = self._sharp._estimate_sharpness(clahe_dict)
        sharp_duration = time.time() - sharp_start
        logger.info(f"Estimación de nitidez completada en {sharp_duration:.3f} segundos")
        
        polygons_to_bin = polygons_to_bin
        
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
        refined_polygons = self._fragment._intercept_polygons(binarized_polygons, sharp_dict)
        step_duration = time.time() - step_start
                
        pipeline_end = time.time() - pipeline_start
        logger.info(f"GENERACIÓN DE POLIGONAL COMPLETADA en {pipeline_end:.4f}s")
        
        time_poly = pipeline_end
        
        return refined_polygons, time_poly

    def _save_problematic_polygons(self, all_polygons: Dict[str, Any], image_name: str) -> None:
        """Guarda imágenes de polígonos problemáticos si está habilitado."""
        if not self.manager_config.get('output_flag', {}).get('problematic_ids', False):
            return
        output_folder = self.manager_config.get('output_folder')
            
        problematic_ids = self._fragment._get_problematic_ids()
        if not problematic_ids:
            return
            
        output_folder = self.preprocessing_config.get('output_folder')
        if not output_folder:
            logger.warning("No se puede guardar polígonos problemáticos porque 'output_folder' no está definido.")
            return
            
        try:
            os.makedirs(output_folder, exist_ok=True)
            saved_count = 0
            for i, (poly_id, poly_data) in enumerate(all_polygons.items()):
                if poly_id in problematic_ids:
                    img = poly_data.get('cropped_img')
                    if img is not None:
                        img_filename = f"{image_name}_problematic_{i+1}.png"
                        img_path = os.path.join(output_folder, img_filename)
                        cv2.imwrite(img_path, img)
                        saved_count += 1
            logger.info(f"PreprocessingManager: Guardadas {saved_count} imágenes problemáticas en {output_folder}")
        except Exception as e:
            logger.error(f"PreprocessingManager: Error guardando imágenes problemáticas: {e}")