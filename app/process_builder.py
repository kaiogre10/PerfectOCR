# PerfectOCR/app/process_builder.py
import logging
from typing import Optional, Dict, Any, Tuple, List
from core.domain.data_formatter import DataFormatter
import services.storage_service as storage_service

logger = logging.getLogger(__name__)

class ProcessingBuilder:
    __slots__ = (
        "all_stagers",
        "logs_config",
        "time_stages_log",
        "time_worker_log"
    )
    """Director de Operaciones: Recibe a sus Jefes de Área ya entrenados y coordina el procesamiento técnico de una sola imagen."""
    def __init__(self, all_stagers: List[Any], logs_config: Dict[str, Any]):
        self.all_stagers = all_stagers
        self.logs_config = logs_config
        self.time_stages_log = logs_config.get("time_stages_log")
        self.time_worker_log = logs_config.get("time_worker_log")
        
    def process_single_image(self, image_data: Dict[str, Any]) -> Optional[Tuple[int, List[int]]]:
        """
        Procesa una sola imagen usando el método execute() uniforme de cada stager.
        Recibe image_data para configurar el contexto de esta ejecución específica.
        Devuelve Direcciones en memoria de los datos generados
        """
        try:    
            # Crear instancia fresca de DataFormatter para esta imagen
            manager = DataFormatter(self.logs_config)
            # Crear contexto para esta ejecución
            context: Dict[str, Any] = {}
            context = {
                "image_data": image_data,
                "time_worker_log": self.time_worker_log
            }

            for _, stager in enumerate(self.all_stagers):
                manager, time_poly = stager.execute(manager, context)
                if manager is None:
                    return None
                
                if self.time_stages_log:
                    logger.info(f"Fase de preparación completada en: {time_poly:.6f}s")

            img_results = manager.get_final_data()
            if img_results.empty:
                return None
            
            if storage_service.storage_data(img_results):
                return (1, [1])
            else:
                return None
            
        except Exception as e:
            logger.error(f"Error fatal procesando la imagen: '{e}'", exc_info=True)
        return None
