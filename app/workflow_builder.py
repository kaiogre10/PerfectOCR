# PerfectOCR/management/workflow_manager.py
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils.batch_tools import chunked, get_optimal_workers, estimate_processing_time
from services.config_service import ConfigService

logger = logging.getLogger(__name__)

class WorkBuilder:
    """
    Director de Logística: Planifica, cuenta y reporta a Main.
    HIPER-ESPECIALIZADO en: contar imágenes, decidir modo, generar reporte.
    NO procesa imágenes - solo planifica.
    """
    def __init__(self, config_manager: ConfigService, project_root: str):
        self.config_manager = config_manager
        self.project_root = project_root
        self.processing_config = config_manager._processing_config
        self.batch_config = self.processing_config.get('batch_processing', {})
        self.small_batch_limit = self.batch_config.get('small_batch_limit', 5)
        self.max_physical_cores = config_manager._max_workers_for_cpu

    def _count_and_plan(self) -> Dict[str, Any]:
        """
        PLANIFICA el procesamiento: cuenta imágenes y decide estrategia.
        REPORTA a Main: cuántos builders crear y qué modo usar.
        """
        paths_config = self.config_manager._paths_config
        input_folder = paths_config.get('input_folder', "")
        valid_extensions = self._get_valid_extensions()

        if not os.path.isdir(input_folder):
            logger.critical(f"La carpeta de entrada no existe: {input_folder}")
            return {"error": f"Carpeta de entrada no encontrada: {input_folder}"}

        image_paths = self._extract_valid_image_paths(input_folder, valid_extensions)
        
        if not image_paths:
            logger.critical("No se encontraron imágenes válidas en la carpeta de entrada.")
            return {"error": "No se encontraron imágenes válidas"}

        # DECIDIR modo de procesamiento
        num_images = len(image_paths)
        use_batch = num_images > self.small_batch_limit
        mode = 'batch' if use_batch else 'interactive'
        
        # CALCULAR workers necesarios
        if use_batch:
            workers_needed = get_optimal_workers(num_images, self.max_physical_cores)
        else:
            workers_needed = 1

        # GENERAR reporte para Main
        estimation = estimate_processing_time(num_images, self.max_physical_cores)
        
        logger.info(f"PLANIFICACIÓN COMPLETADA:")
        logger.info(f"  - {num_images} imágenes detectadas")
        logger.info(f"  - Modo: {mode.upper()}")
        logger.info(f"  - Workers requeridos: {workers_needed}")
        logger.info(f"  - Tiempo estimado: {estimation['parallel_minutes']:.1f} min")

        return {
            "status": "success",
            "total_images": num_images,
            "mode": mode,
            "workers_needed": workers_needed,
            "image_paths": image_paths,  # Lista completa de rutas específicas
            "processing_estimation": estimation
        }

    def _extract_valid_image_paths(self, input_folder: str, valid_extensions: tuple) -> List[str]:
        """Extrae lista de rutas completas de imágenes válidas."""
        image_paths = []
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(valid_extensions):
                full_path = os.path.join(input_folder, filename)
                image_paths.append(full_path)
        return image_paths

    def _get_valid_extensions(self) -> tuple:
        """Obtiene extensiones válidas desde configuración."""
        image_loader_config = self.config_manager._image_loader_config
        extensions = image_loader_config.get('image_loader', {}).get('extensions', {}).get('valid_image_extensions', [])
        return tuple(extensions)
