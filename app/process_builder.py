# PerfectOCR/app/process_builder.py
import logging
from typing import Optional, Dict, Any, List
from core.domain.data_formatter import DataFormatter
from services.storage_service import storage_data
from services.output_service import write_temp_log

logger = logging.getLogger(__name__)

class ProcessingBuilder:
    __slots__ = (
        "all_stagers",
        "logs_debug",
        "handle_storage"
    )
    """Director de Operaciones: Recibe a sus Jefes de Área ya entrenados y coordina el procesamiento técnico de una sola imagen."""
    def __init__(self, all_stagers: List[Any], logs_debug: Dict[str, Any], handle_storage: bool):
        self.all_stagers = all_stagers
        self.logs_debug = logs_debug
        self.handle_storage = handle_storage
        
    def process_single_image(self, image_data: Dict[str, Any]) -> Optional[List[int]]:
        """
        Procesa una sola imagen usando el método execute() uniforme de cada stager.
        Recibe image_data para configurar el contexto de esta ejecución específica.
        Devuelve Direcciones en memoria de los datos generados
        """
        try:
            # Crear instancia fresca de DataFormatter para esta imagen
            manager = DataFormatter(self.logs_debug)
            # Crear contexto para esta ejecución
            context: Dict[str, Any] = {}
            context = {
                "image_data": image_data,
                "time_worker_log": self.logs_debug.get("time_worker_log")
            }

            for _, stager in enumerate(self.all_stagers):
                manager, time_poly = stager.execute(manager, context)
                if manager is None:
                    return None
                
                if self.logs_debug.get("time_stages_log"):
                    logger.info(f"Fase de preparación completada en: {time_poly:.6f}s")

            name = manager.payload.name if manager.payload else None
            plain_text = manager.payload.payload if manager.payload else None
            buffer_size = manager.payload.buffer_sizes if manager.payload else None

            if plain_text is None or buffer_size is None or name is None:
                return None
            
            if self.handle_storage and storage_data(plain_text, buffer_size):
                write_temp_log((name, plain_text))
                return buffer_size
            else:
                return None
            
        except Exception as e:
            logger.error(f"Error fatal procesando la imagen: '{e}'", exc_info=True)
        return None
