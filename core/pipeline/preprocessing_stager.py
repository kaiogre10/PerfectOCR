# PerfectOCR/core/coordinators/preprocessing_coordinator.py
import logging
import time
import numpy as np
from typing import Any, Dict, Tuple, List, Optional
from core.domain.data_formatter import DataFormatter
from core.factory.abstract_worker import PreprossesingAbstractWorker
from services.output_service import OutputService
from core.domain.data_models import CroppedImage

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
        """Ejecuta secuencialmente cada worker sobre todas las imágenes recortadas.

        Flujo:
          1. Obtiene diccionario {poly_id: CroppedImage}
          2. Para cada worker procesa todos los polígonos
          3. Almacena temporalmente resultados binarizados para inyectarlos en fragmentator
          4. Actualiza el manager al final
        """
        start_time = time.time()
        logger.info("[PreprocessingManager] Iniciando pipeline secuencial directo")
        
        cropped_images: Dict[str, CroppedImage] = manager.get_cropped_images_for_preprocessing()
        
        cropped_img: CroppedImage = cropped_images.get("cropped_img")
        for worker_idx, worker in enumerate(self.workers):
            worker_name = worker.__class__.__name__
            logger.info(f"[PreprocessingManager] Worker {worker_idx + 1}/{len(self.workers)}: {worker_name}")

            cropped_img = worker.preprocess(cropped_img, manager)
            if not cropped_img:
                logger.error(f"[{worker_name}] Fallo procesando polígono {poly_id}")
                continue
                    
        # Actualizar estado final en manager
        for poly_id, poly_data in cropped_images.items():
            try:
                manager.update_preprocessing_result(poly_id, cropped_img, worker_name, True)
            except Exception as e:
                logger.error(f"Error actualizando resultado de polígono {poly_id}: {e}")

        elapsed = time.time() - start_time
        logger.info(f"[PreprocessingStager] Pipeline secuencial completado en: {elapsed:.3f}s; polígonos: {len(cropped_images)}")
        return manager, elapsed