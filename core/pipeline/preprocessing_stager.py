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
        """Ejecuta secuencialmente cada worker sobre todas las imágenes recortadas.

        Flujo:
          1. Obtiene diccionario {poly_id: CroppedImage}
          2. Para cada worker procesa todos los polígonos
          3. Almacena temporalmente resultados binarizados para inyectarlos en fragmentator
          4. Actualiza el manager al final
        """
        start_time = time.time()
        logger.info("[PreprocessingManager] Iniciando pipeline secuencial directo")
        for worker_idx, worker in enumerate(self.workers):
            worker_name = worker.__class__.__name__
            logger.info(f"[PreprocessingManager] Worker {worker_idx + 1}/{len(self.workers)}: {worker_name}")

        metadata = manager.get_metadata
        cropped_img = manager.get_polygons_with_cropped_img
        
        context: Dict[str, Any] = {
            "metadata": metadata,
            "cropped_img": cropped_img            
        }
        for worker in self.workers:
            if not worker.preprocess(context, manager):
                logger.error(f"InputStager: Fallo en el worker {worker.__class__.__name__}")
                return None, 0.0

        elapsed = time.time() - start_time
        logger.info(f"[PreprocessingStager] Pipeline secuencial completado en: {elapsed:.3f}s; polígonos: {len(cropped_images)}")
        return manager, elapsed