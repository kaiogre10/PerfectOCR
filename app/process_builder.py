# PerfectOCR/app/process_builder.py
import logging
import time
from typing import Optional, Dict, Any, List, Tuple
from domain.data_formatter import DataFormatter
from services.storage_service import storage_data
from services.output_service import write_temp_log

logger = logging.getLogger(__name__)

class ProcessingBuilder:
    __slots__ = (
        "project_root",
        "all_stagers",
        "logs_debug"
    )
    """Director de Operaciones: Recibe a sus Jefes de Área ya entrenados y coordina el procesamiento técnico de una sola imagen."""
    def __init__(self, project_root: str, all_stagers: List[Any], logs_debug: Dict[str, Any]):
        self.project_root = project_root
        self.all_stagers = all_stagers
        self.logs_debug = logs_debug
        
    def process_single_image(self, image_path: str) -> Optional[Tuple[str, int]]:
        """
        Procesa una sola imagen usando el método execute() uniforme de cada stager.
        Devuelve Direcciones en memoria de los datos generados
        """
        try:
            # Crear instancia fresca de DataFormatter para esta imagen
            manager = DataFormatter(self.logs_debug)
            # Crear contexto para esta ejecución
            context: Dict[str, Any] = {
                "image_data": image_path,
                "time_worker_log": self.logs_debug.get("time_worker_log")
            }
            try:
                for _, stager in enumerate(self.all_stagers):
                    stager_name = stager.__class__.__name__
                    stager_time = time.perf_counter()
                    manager = stager.execute(manager, context)
                    if manager is None:
                        logger.error(f"Falla en '{stager_name}', tiempo: {time.perf_counter() - stager_time}'s", exc_info=True)
                        del context
                        return None
                
                    if self.logs_debug.get("time_stages_log"):
                        logger.info(f"Fase de preparación completada en: {time.perf_counter() - stager_time:.6f}'s")
            except RuntimeError as e:
                logger.error(f"ERROR PROCESANDO: '{e}'", exc_info=True)
                del context
                return None

            name = manager.payload.name if manager.payload else None
            plain_text = manager.payload.payload if manager.payload else None

            if plain_text is None or name is None:
                if manager is None:
                    logger.info(f"ERROR")
                    del context
                    return None
                else:
                    manager.reset_data()
                    logger.info(f"Proceso correcto con early return, se deuelve datos MOCK para debug")
                    del context
                    return "", 0
                
            elif write_temp_log((name, plain_text)):
                if self.logs_debug.get("handle_memory"):
                    buff_size = storage_data(plain_text)
                    if buff_size is not None:
                        logger.warning(f"PAYLOAD GUARDADO EN MEMORIA: '{buff_size} B', Y EN ARCHIVO DE SEGURIDAD")
                        del context
                        return name, buff_size
                    
                    manager.reset_data()
                    del context
                    return None
                
                else:
                    logger.info("NO SE ACTIVO MEMORIA DINÁMICA, SE REGRESAN LOS BYTES ESTIMADOS SOLAMENTE")
                    manager.reset_data()
                    del context
                    return name, (len(plain_text.encode("ascii", 'ignore')) * 2)
            else:
                logger.error("NO SE PUDO GENERAR ARCHIVO DE SEGURIDAD")
                manager.reset_data()
                del context
                return None
            
        except Exception as e:
            logger.error(f"Error fatal procesando la imagen: '{e}'", exc_info=True)
        return None
