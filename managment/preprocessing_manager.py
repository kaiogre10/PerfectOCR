# PerfectOCR/core/coordinators/preprocessing_coordinator.py
import cv2
import numpy as np
import logging
import time
import os
from typing import Any, Optional, Dict, Tuple, List
from core.preprocessing.moire import MoireDenoiser
from core.preprocessing.sp import DoctorSaltPepper
from core.preprocessing.gauss import GaussianDenoiser
from core.preprocessing.clahe import ClaherEnhancer
from core.preprocessing.sharp import SharpeningEnhancer
#from core.utils.output_handlers import ImageOutputHandler

logger = logging.getLogger(__name__)

class PreprocessingManager:
    """
    Coordina la fase de preprocesamiento, delegando todo el trabajo a un único worker autosuficiente.
    """
    def __init__(self, config: Dict, project_root: str):
        self.project_root = project_root
        self.workflow_config = config.get('workflow', {})
        self.output_config = config.get('output_config', {})
        quality_rules = config.get('corrections', {})
        
        # Instanciar workers con sus configuraciones específicas
        self._moire = MoireDenoiser(config=quality_rules.get('denoise', {}), project_root=self.project_root)
        self._sp = DoctorSaltPepper(config=quality_rules.get('denoise', {}), project_root=self.project_root)
        self._gauss = GaussianDenoiser(config=quality_rules.get('denoise', {}), project_root=self.project_root)
        self._claher = ClaherEnhancer(config=quality_rules.get('contrast', {}), project_root=self.project_root)
        self._sharp = SharpeningEnhancer(config=quality_rules.get('sharpening', {}), project_root=self.project_root)
#        self.image_saver = ImageOutputHandler()
        
    def _apply_preprocessing_pipelines(self, refined_polygons: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], float]:         
        """
        Procesa la imagen de forma secuencial:
        1. Moiré
        2. Ruido sal y pimienta
        3. Ruido general
        4. Contraste
        5. Nitidez
        """
        pipeline_start = time.time()

        # 1. Remoción de moiré
        moire_img = self._moire._detect_moire_patterns(refined_polygons)
        # 2. Filtro de ruido sal y pimienta
        sp_img = self._sp._estimate_salt_pepper_noise(moire_img)
        # 3. Filtro de ruido general
        gauss_img = self._gauss._estimate_gaussian_noise(sp_img)
        # 4. Mejora de contraste
        clahed_img = self._claher._estimate_contrast(gauss_img)
        # 5. Mejora de nitidez
        corrected_imag = self._sharp._estimate_sharpness(clahed_img)
            
        preprocessed_polygons = corrected_imag
        total_duration = time.time() - pipeline_start
        return preprocessed_polygons, total_duration

