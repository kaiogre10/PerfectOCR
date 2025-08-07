# PerfectOCR/management/workflow_manager.py
import os
import logging
from typing import List, Dict, Any, Tuple
from utils.batch_tools import get_optimal_workers
from services.config_service import ConfigService

logger = logging.getLogger(__name__)

class WorkFlowBuilder:
    """
    Director de Logística: Planifica, cuenta y reporta a Main.
    HIPER-ESPECIALIZADO en: contar imágenes, decidir modo, generar reporte.
    NO procesa imágenes - solo planifica.
    """
    def __init__(self, config_services: ConfigService, project_root: str):
        # SOLO recibe ConfigService - NO necesita input_paths externos
        self.config_services = config_services
        self.project_root = project_root
        self.processing_config = config_services.processing_config
        self.batch_config = self.processing_config.get('batch_processing', {})
        self.small_batch_limit = self.batch_config.get('small_batch_limit', 5)
        self.max_physical_cores = config_services.max_workers_for_cpu

    def count_and_plan(self) -> Dict[str, Any]:
        """
        PLANIFICA el procesamiento: cuenta imágenes y decide estrategia.
        REPORTA a Main: cuántos builders crear y qué modo usar.
        """
        paths_config = self.config_services.paths_config
        input_folder = paths_config.get('input_folder', "")
        valid_extensions = self.get_valid_extensions()

        if not os.path.isdir(input_folder):
            logger.critical(f"La carpeta de entrada no existe: {input_folder}")
            return {"error": f"Carpeta de entrada no encontrada: {input_folder}"}

        image_info = self.extract_valid_image_paths(input_folder, valid_extensions)
        
        if not image_info:
            logger.critical("No se encontraron imágenes válidas en la carpeta de entrada.")
            return {"error": "No se encontraron imágenes válidas"}

        # DECIDIR modo de procesamiento
        num_images = len(image_info)
        use_batch = num_images > self.small_batch_limit
        mode = 'batch' if use_batch else 'interactive'
        
        # CALCULAR workers necesarios
        if use_batch:
            workers_needed = get_optimal_workers(num_images, self.max_physical_cores)
        else:
            workers_needed = 1
        
        
        logger.info(f"PLANIFICACIÓN COMPLETADA:")
        logger.info(f" {num_images} imágenes detectadas")

        workflow_report =  {
            "status": "success",
            "total_images": num_images,
            "mode": mode,
            "workers_needed": workers_needed,
            "image_info": image_info
        }

        return workflow_report
        
    def extract_valid_image_paths(self, input_folder: str, valid_extensions: Tuple[str, ...]) -> List[Dict[str, str]]:
        """Extrae lista de rutas y nombres de imágenes válidas."""
        image_info: List[Dict[str, str]] = []
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(valid_extensions):
                full_path = os.path.join(input_folder, filename)
                image_name = os.path.splitext(filename)[0]
                image_extension = os.path.splitext(filename)[1]
                image_info.append({
                    "path": full_path,
                    "name": image_name,
                    "extension": image_extension
                })
        return image_info

    def get_valid_extensions(self) -> Tuple[str, ...]:
        """Obtiene extensiones válidas desde configuración."""
        extensions = self.config_services.processing_config.get('valid_image_extensions', [])
        return tuple(extensions)
