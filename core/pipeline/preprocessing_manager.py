# PerfectOCR/core/coordinators/preprocessing_coordinator.py
import cv2
import numpy as np
import logging
import time
import os
from typing import Any, Optional, Dict, Tuple, List
from core.workers.preprocessing.moire import MoireDenoiser
from core.workers.preprocessing.sp import DoctorSaltPepper
from core.workers.preprocessing.gauss import GaussianDenoiser
from core.workers.preprocessing.clahe import ClaherEnhancer
from core.workers.preprocessing.sharp import SharpeningEnhancer
#from core.utils.output_handlers import ImageOutputHandler

logger = logging.getLogger(__name__)

class PreprocessingManager:
    """
    Coordina la fase de preprocesamiento, delegando todo el trabajo a un único worker autosuficiente.
    """
    def __init__(self, config: Dict, project_root: str):
        self.project_root = project_root
        # El manager ahora recibe directamente su sección del config.
        self.preprocessing_config = config
        
        # Inyección en cascada a los workers de preprocesamiento:
        denoise_config = self.preprocessing_config.get('denoise', {})
        self._moire = MoireDenoiser(config=denoise_config.get('moire', {}), project_root=self.project_root)
        self._sp = DoctorSaltPepper(config=denoise_config.get('median_filter', {}), project_root=self.project_root)
        self._gauss = GaussianDenoiser(config=denoise_config.get('bilateral_params', {}), project_root=self.project_root)
        self._claher = ClaherEnhancer(config=self.preprocessing_config.get('contrast', {}), project_root=self.project_root)
        self._sharp = SharpeningEnhancer(config=self.preprocessing_config.get('sharpening', {}), project_root=self.project_root)
        
    def _apply_preprocessing_pipelines(self, refined_polygons: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], float]:
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
        polygons = refined_polygons.get("polygons", {})
        if not polygons:
            logger.warning("No se encontraron polígonos para preprocesar")
            return refined_polygons, 0.0
        
        logger.info(f"Procesando {len(polygons)} polígonos")

        # Etapa 1: Detección de patrones Moiré
        moire_start = time.time()
        logger.info("Iniciando detección de patrones Moiré")
        moire_dict = self._moire._detect_moire_patterns(refined_polygons)
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

        total_duration = time.time() - pipeline_start
                
        preprocess_dict = sharp_dict
        return preprocess_dict, total_duration

