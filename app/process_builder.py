# PerfectOCR/app/process_builder.py
import logging
from typing import Optional, Dict, Any, List
from core.domain.data_formatter import DataFormatter
from services.storage_service import storage_data, transform_data

logger = logging.getLogger(__name__)

class ProcessingBuilder:
    __slots__ = (
        "all_stagers",
        "processing_config"
    )
    """Director de Operaciones: Recibe a sus Jefes de Área ya entrenados y coordina el procesamiento técnico de una sola imagen."""
    def __init__(self, all_stagers: List[Any], processing_config: Dict[str, Any]):
        self.all_stagers = all_stagers
        self.processing_config = processing_config
        
    def process_single_image(self, image_data: Dict[str, Any]) -> Optional[List[int]]:
        """
        Procesa una sola imagen usando el método execute() uniforme de cada stager.
        Recibe image_data para configurar el contexto de esta ejecución específica.
        Devuelve Direcciones en memoria de los datos generados
        """
        runtime_config_logs = self.processing_config
        try:
            # Crear instancia fresca de DataFormatter para esta imagen
            manager = DataFormatter(runtime_config_logs)
            # Crear contexto para esta ejecución
            context: Dict[str, Any] = {}
            context = {
                "image_data": image_data,
                "time_worker_log": runtime_config_logs.get("time_worker_log")
            }

            for _, stager in enumerate(self.all_stagers):
                manager, time_poly = stager.execute(manager, context)
                if manager is None:
                    return None
                
                if runtime_config_logs.get("time_stages_log"):
                    logger.info(f"Fase de preparación completada en: {time_poly:.6f}s")

            img_results = manager.get_final_data()
            if img_results.empty:
                return None
            
            # if runtime_config_logs.get("handle_memory"):
            sizes = storage_data(img_results)
            if sizes and sum(sizes) > 1:
                return sizes
            else:
                return None
            # else:
            #     sizes = transform_data(img_results)[0]
            #     return None if not sizes else sizes
            
        except Exception as e:
            logger.error(f"Error fatal procesando la imagen: '{e}'", exc_info=True)
        return None
