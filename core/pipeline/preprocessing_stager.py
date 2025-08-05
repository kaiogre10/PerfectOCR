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
from core.domain.workflow_job import WorkflowJob, ProcessingStage

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
        
    def _apply_preprocessing_pipelines(self, workflow_job: WorkflowJob, polygons_to_bin: Dict[str, Any]) -> Tuple[Optional[WorkflowJob], float]:
        """
        Procesa el WorkflowJob de forma secuencial, modificando los polígonos.
        """
        pipeline_start = time.time()
        logger.info("[PreprocessingManager] Iniciando pipeline de preprocesamiento")

        if not workflow_job.polygons:
            logger.warning("[PreprocessingManager] No se encontraron polígonos para preprocesar")
            return workflow_job, 0.0
        
        # Crear un diccionario temporal para compatibilidad con workers existentes
        polygons_dict = {pid: p.cropped_img for pid, p in workflow_job.polygons.items() if p.cropped_img is not None}
        
        # Etapa 1: Detección de patrones Moiré
        temp_dict_for_processing = {"polygons": {pid: {"cropped_img": img} for pid, img in polygons_dict.items()}}
        moire_dict = self._moire._detect_moire_patterns(temp_dict_for_processing)

        # Etapa 2: Estimación de ruido sal y pimienta
        sp_dict = self._sp._estimate_salt_pepper_noise(moire_dict)
        
        # Etapa 3: Estimación de ruido gaussiano
        gauss_dict = self._gauss._estimate_gaussian_noise(sp_dict)
        
        # Etapa 4: Estimación de contraste
        clahe_dict = self._claher._estimate_contrast(gauss_dict)

        # Etapa 5: Estimación de nitidez
        sharp_dict = self._sharp._estimate_sharpness(clahe_dict)
        
        # Actualizar las imágenes 'cropped_img' en el WorkflowJob
        for poly_id, poly_data in sharp_dict.get("polygons", {}).items():
            if poly_id in workflow_job.polygons and "cropped_img" in poly_data:
                workflow_job.polygons[poly_id].cropped_img = poly_data["cropped_img"]

        # Binarización
        binarized_polygons = self._bin._binarize_polygons(polygons_to_bin)
        if binarized_polygons:
             for poly_id, bin_img in binarized_polygons.items():
                if poly_id in workflow_job.polygons:
                    workflow_job.polygons[poly_id].bin_img = bin_img

        # Fragmentación
        refined_polygons = self._fragment._intercept_polygons(binarized_polygons, sharp_dict.get("polygons", {}))

        # Actualizar el workflow_job con los polígonos refinados
        for poly_id, poly_data in refined_polygons.items():
            if poly_id in workflow_job.polygons:
                workflow_job.polygons[poly_id].was_fragmented = poly_data.get("was_fragmented", False)
                # Actualiza otros campos si es necesario
        
        workflow_job.update_stage(ProcessingStage.PREPROCESSING_COMPLETE)
        
        total_time = time.time() - pipeline_start
        workflow_job.processing_times["preprocessing"] = total_time
        
        logger.info(f"[PreprocessingManager] Preprocesamiento completado en {total_time:.3f}s")
        return workflow_job, total_time

    def _save_problematic_polygons(self, workflow_job: WorkflowJob, image_name: str) -> None:
        """Guarda imágenes de polígonos problemáticos si está habilitado."""
        if not self.manager_config.get('output_flag', {}).get('problematic_ids', False):
            return
            
        problematic_ids = self._fragment._get_problematic_ids()
        if not problematic_ids:
            return
            
        output_folder = self.preprocessing_config.get('output_folder')
        if not output_folder:
            logger.warning("[PreprocessingManager] No se puede guardar polígonos problemáticos porque 'output_folder' no está definido.")
            return
            
        try:
            os.makedirs(output_folder, exist_ok=True)
            saved_count = 0
            for poly_id in problematic_ids:
                if poly_id in workflow_job.polygons:
                    img = workflow_job.polygons[poly_id].cropped_img
                    if img is not None:
                        img_filename = f"{image_name}_problematic_{poly_id}.png"
                        img_path = os.path.join(output_folder, img_filename)
                        cv2.imwrite(img_path, img)
                        saved_count += 1
            logger.info(f"[PreprocessingManager] Guardadas {saved_count} imágenes problemáticas en {output_folder}")
        except Exception as e:
            logger.error(f"[PreprocessingManager] Error guardando imágenes problemáticas: {e}")
