# PerfectOCR/core/coordinators/preprocessing_coordinator.py
import logging
import time
from typing import Any, Dict, Tuple, List, Optional
from core.domain.data_formatter import DataFormatter
from core.factory.abstract_worker import PreprossesingAbstractWorker
from services.output_service import OutputService

logger = logging.getLogger(__name__)

class PreprocessingStager:
    """
    Coordina la fase de preprocesamiento, delegando todo el trabajo a un único worker autosuficiente.
    """
    def __init__(self, workers: List[PreprossesingAbstractWorker], stage_config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.workers = workers
        self.stage_config = stage_config
        self.output_flags = self.stage_config.get("output_flag", "")
        self.output_folder = self.stage_config["output_folder"]
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
        
    def apply_preprocessing_pipelines(self, manager: DataFormatter) -> Tuple[Optional[DataFormatter], float]:
        start_time = time.time()
        logger.info("[PreprocessingManager] Iniciando pipeline secuencial directo")
        
        cropped_images = manager.get_cropped_images_for_preprocessing()
        if not cropped_images:
            logger.warning("[PreprocessingManager] No hay imágenes para procesar")
            return manager, 0.0
        
        # Variable temporal para almacenar imágenes binarizadas
        binarized_images = {}
        
        for worker_idx, worker in enumerate(self.workers):
            worker_name = worker.__class__.__name__
            logger.info(f"[PreprocessingManager] Worker {worker_idx + 1}/{len(self.workers)}: {worker_name}")
            
            processed_images = {}
            
            for poly_id, cropped_img in cropped_images.items():
                try:
                    # Worker procesa la imagen
                    processed_img = worker.preprocess(cropped_img, manager)
                    processed_images[poly_id] = processed_img
                    
                    # Si es el binarizador, guardar las imágenes binarizadas temporalmente
                    if worker_name == "binarization":
                        binarized_images[poly_id] = processed_img.cropped_img
                    
                    logger.debug(f"[{worker_name}] Polígono {poly_id} procesado exitosamente")
                    
                except Exception as e:
                    logger.error(f"[{worker_name}] Error procesando polígono {poly_id}: {e}", exc_info=True)
                    processed_images[poly_id] = cropped_img
            
            # Pasar las imágenes procesadas al siguiente worker
            cropped_images = processed_images
            
            # Si el siguiente worker es el fragmentador, inyectar las imágenes binarizadas
            if worker_idx + 1 < len(self.workers):
                next_worker = self.workers[worker_idx + 1]
                if next_worker.__class__.__name__ == "fragmentator":
                    # Inyectar las imágenes binarizadas en el contexto del fragmentador
                    next_worker.set_binarized_images(binarized_images)
            
            logger.info(f"[{worker_name}] Completado. {len(processed_images)} imágenes procesadas")
        
        # Actualizar el workflow
        for poly_id, final_img in cropped_images.items():
            manager.update_preprocessing_result(poly_id, final_img, "pipeline_complete", True)
        
        total_time = time.time() - start_time
        logger.info(f"[PreprocessingStager] Pipeline secuencial completado en: {total_time:.3f}s")
        return manager, total_time