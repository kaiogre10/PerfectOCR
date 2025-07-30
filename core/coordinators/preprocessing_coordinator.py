# PerfectOCR/core/coordinators/preprocessing_coordinator.py
import cv2
import numpy as np
import logging
import time
import os
from typing import Any, Optional, Dict, Tuple, List
from core.workflow.preprocessing.moire import MoireDenoiser
from core.workflow.preprocessing.sp import DoctorSaltPepper
from core.workflow.preprocessing.gauss import GaussianDenoiser
from core.workflow.preprocessing.clahe import ClaherEnhancer
from core.workflow.preprocessing.sharp import SharpeningEnhancer

from core.workspace.utils.output_handlers import ImageOutputHandler

logger = logging.getLogger(__name__)

class PreprocessingCoordinator:
    """
    Coordina la fase de preprocesamiento, delegando todo el trabajo
    a un único worker autosuficiente.
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
        self.image_saver = ImageOutputHandler()
        
    def _apply_preprocessing_pipelines(
        self,
        refined_polygons: List[Dict[str, Any]],
        input_path: str = ""
    ) -> Tuple[Dict[str, Any], float]:
        """
        Procesa la imagen de forma secuencial:
        1. Moiré
        2. Ruido sal y pimienta
        3. Ruido general
        4. Contraste
        5. Nitidez
        Y guarda la imagen de cada polígono si está habilitado en la configuración.
        """
        polygons_list = refined_polygons
        polygons_received = len(polygons_list)
        polygons_corrected = 0
        images_saved_count = 0
        
        pipeline_start = time.time()
        
        base_name = os.path.splitext(os.path.basename(input_path))[0] if input_path else "unknown_doc"
        should_save_images = self.output_config.get('enabled_outputs', {}).get('preprocessed_image', False)
        output_folder = self.workflow_config.get('output_folder')

        for i, polygon in enumerate(polygons_list):
            cropped_img = polygon.get("cropped_img")
            if cropped_img is not None:
        
                # 1. Remoción de moiré
                moire_img = self._moire._detect_moire_patterns(cropped_img)
                
                # 2. Filtro de ruido sal y pimienta
                sp_img = self._sp._estimate_salt_pepper_noise(moire_img)
                
                # 3. Filtro de ruido general
                gauss_img = self._gauss._estimate_gaussian_noise(sp_img)
                
                # 4. Mejora de contraste
                clahed_img = self._claher._estimate_contrast(gauss_img)
            
                # 5. Mejora de nitidez
                corrected_image = self._sharp._estimate_sharpness(clahed_img)
                
                # Actualizar el polígono con la imagen procesada
                polygon["processed_img"] = corrected_image
                polygons_corrected += 1

                # Guardar la imagen preprocesada de este polígono si está habilitado
                if should_save_images:
                    if output_folder:
                        polygon_id = polygon.get('polygon_id', f'poly_{i}')
                        output_filename = f"{base_name}_preprocessed_{polygon_id}.png"
                        if self.image_saver.save(corrected_image, output_folder, output_filename):
                            images_saved_count += 1
                    elif i == 0:  # Loguear el error solo una vez
                        logger.error("La 'output_folder' no está definida, no se guardarán imágenes preprocesadas.")

        total_duration = time.time() - pipeline_start
        
        logger.info(f"Preprocesamiento - Polígonos recibidos: {polygons_received}, Polígonos corregidos: {polygons_corrected}")
        if images_saved_count > 0:
            logger.info(f"Guardadas {images_saved_count} imágenes preprocesadas en '{output_folder}'.")

        try:
            return {"polygons": polygons_list}, total_duration
        except Exception as e:
            logger.error(f"Preprocessing: Falló el pipeline del worker: {e}", exc_info=True)
            return {"polygons": []}, 0.0