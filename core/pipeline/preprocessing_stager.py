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
from services.output_service import OutputService

logger = logging.getLogger(__name__)

class PreprocessingStager:
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
        self._bin = Binarizator(config=self.preprocessing_config.get('binarize', {}), project_root=self.project_root)
        self._fragment = PolygonFragmentator(config=self.preprocessing_config.get('fragmentation', {}), project_root=self.project_root)

        self.output_flags = self.manager_config.get("output_flag", {})
        self.output_folder = self.manager_config["output_folder"]
        self.output_service = None
        if self.output_folder and any([
            self.output_flags.get("moire_poly", False),
            self.output_flags.get("sp_poly", False),
            self.output_flags.get("gauss_poly", False),
            self.output_flags.get("clahe_poly", False),
            self.output_flags.get("sharp_poly", False),
            self.output_flags.get("binarized_polygons", False),
            self.output_flags.get("refined_polygons", False),
            self.output_flags.get("problematic_polygons", False)
        ]):
            self.output_service = OutputService()
        
    def apply_preprocessing_pipelines(
        self, 
        workflow_job: WorkflowJob
        ) -> Tuple[Optional[WorkflowJob], float]:
        """
        Procesa el WorkflowJob de forma secuencial, modificando los polígonos in-situ
        siguiendo una filosofía de pipeline limpio.
        """
        pipeline_start = time.time()
        logger.info("[PreprocessingManager] Iniciando pipeline de preprocesamiento")

        if not workflow_job.polygons:
            logger.warning("[PreprocessingManager] No se encontraron polígonos para preprocesar")
            return workflow_job, 0.0
        
        # Crear un único diccionario de trabajo que se modificará en cada etapa.
        processing_dict = {
            "polygons": {
                pid: {"cropped_img": p.cropped_img} 
                for pid, p in workflow_job.polygons.items() 
                if p.cropped_img is not None
            }
        }
        image_name = workflow_job.doc_metadata.doc_name if workflow_job.doc_metadata else "document"
        
        processing_dict = self._moire._detect_moire_patterns(processing_dict)
        if self.output_service and self.output_flags.get("moire_poly", False):
            moire_imgs = [d["cropped_img"] for d in processing_dict.get("polygons", {}).values() if d.get("cropped_img") is not None]
            self.output_service.save_images(moire_imgs, self.output_folder, f"{image_name}_moire")

        processing_dict = self._sp._estimate_salt_pepper_noise(processing_dict)
        if self.output_service and self.output_flags.get("sp_poly", False):
            sp_imgs = [d["cropped_img"] for d in processing_dict.get("polygons", {}).values() if d.get("cropped_img") is not None]
            self.output_service.save_images(sp_imgs, self.output_folder, f"{image_name}_sp")

        processing_dict = self._gauss._estimate_gaussian_noise(processing_dict)
        if self.output_service and self.output_flags.get("gauss_poly", False):
            gauss_imgs = [d["cropped_img"] for d in processing_dict.get("polygons", {}).values() if d.get("cropped_img") is not None]
            self.output_service.save_images(gauss_imgs, self.output_folder, f"{image_name}_gauss")

        processing_dict = self._claher._estimate_contrast(processing_dict)
        if self.output_service and self.output_flags.get("clahe_poly", False):
            clahe_imgs = [d["cropped_img"] for d in processing_dict.get("polygons", {}).values() if d.get("cropped_img") is not None]
            self.output_service.save_images(clahe_imgs, self.output_folder, f"{image_name}_clahe")

        processing_dict = self._sharp._estimate_sharpness(processing_dict)
        if self.output_service and self.output_flags.get("sharp_poly", False):
            sharp_imgs = [d["cropped_img"] for d in processing_dict.get("polygons", {}).values() if d.get("cropped_img") is not None]
            self.output_service.save_images(sharp_imgs, self.output_folder, f"{image_name}_sharp")

        # Binarización: Se realiza sobre las imágenes ya procesadas por 'sharp'.
        polygons_to_bin = processing_dict.get("polygons", {})
        binarized_polygons = self._bin._binarize_polygons(polygons_to_bin)
        if self.output_service and self.output_flags.get("binarized_polygons", False):
            bin_imgs = [img for img in binarized_polygons.values() if img is not None]
            self.output_service.save_images(bin_imgs, self.output_folder, f"{image_name}_binarized")

        # Fragmentación: Mide sobre las binarizadas, pero corta sobre las de alta calidad (de 'sharp').
        # El resultado de la fragmentación también actualiza 'processing_dict'.
        processing_dict = self._fragment._intercept_polygons(binarized_polygons, processing_dict)
        if self.output_service and self.output_flags.get("refined_polygons", False):
            refined_imgs = [d.get("cropped_img") for d in processing_dict.get("polygons", {}).values() if d.get("cropped_img") is not None]
            self.output_service.save_images(refined_imgs, self.output_folder, f"{image_name}_refined")

        # Guardar polígonos problemáticos (solo imágenes)
        if self.output_service and self.output_flags.get("problematic_polygons", False):
            self._save_problematic_polygons(workflow_job, image_name)

        # --- Actualización Final y Limpia del WorkflowJob ---
        # Se actualiza el workflow_job una sola vez al final con los resultados definitivos.
        final_polygons_data = processing_dict.get("polygons", {})
        for poly_id, poly_data in final_polygons_data.items():
            if poly_id in workflow_job.polygons:
                # Actualizar la imagen preprocesada final
                if "cropped_img" in poly_data:
                    workflow_job.polygons[poly_id].cropped_img = poly_data["cropped_img"]
                
                # Actualizar el estado de fragmentación
                if "was_fragmented" in poly_data:
                    workflow_job.polygons[poly_id].was_fragmented = poly_data["was_fragmented"]
            else:
                # Aquí agregas los nuevos
                workflow_job.polygons[poly_id] = poly_data
        
        workflow_job.update_stage(ProcessingStage.PREPROCESSING_COMPLETE)
        
        total_time = time.time() - pipeline_start
        workflow_job.processing_times["preprocessing"] = total_time
        
        logger.info(f"[PreprocessingManager] Preprocesamiento completado en {total_time:.3f}s")
        return workflow_job, total_time
        
    def _save_problematic_polygons(self, workflow_job: WorkflowJob, image_name: str) -> None:
        
        """Guarda imágenes de polígonos problemáticos."""
        problematic_ids = self._fragment._get_problematic_ids()
        if not problematic_ids:
            return
            
        problematic_imgs = []
        for poly_id in problematic_ids:
            if poly_id in workflow_job.polygons:
                img = workflow_job.polygons[poly_id].cropped_img
                if img is not None:
                    problematic_imgs.append(img)
        
        if problematic_imgs and self.output_service is not None:
            self.output_service.save_images(problematic_imgs, self.output_folder, f"{image_name}_problematic")
            logger.info(f"[PreprocessingManager] Guardadas {len(problematic_imgs)} imágenes problemáticas")