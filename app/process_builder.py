# PerfectOCR/app/process_builder.py
import logging
import time
from typing import Optional, Dict, Any, List, Tuple
from domain.data_formatter import DataFormatter
from services.output_service import write_temp_log
from domain.class_models import StringsModels
from utils.text_utils import get_ids

logger = logging.getLogger(__name__)

class ProcessingBuilder:
    __slots__ = ("project_root", "all_stagers", "logs_debug", "memory", "time_stages_log")
    """Director de Operaciones: Recibe a sus Jefes de Área ya entrenados y coordina el procesamiento técnico de una sola imagen."""
    def __init__(self, project_root: str, all_stagers: List[Any], logs_debug: Dict[str, Any]):
        self.project_root = project_root
        self.all_stagers = all_stagers
        self.logs_debug = logs_debug
        self.memory = logs_debug.get("handle_memory")
        self.time_stages_log = logs_debug.get("time_stages_log")
        
    def process_query(self, query: List[str]) -> Optional[Tuple[List[str], List[int]]]:
        results = self.process_images(query)
        if not results:
            return None

        names = [text[0] for text in results]
        logger.error(f"IMAGENES EXITOSAS: {names}")
        buff_size = [text[2] for text in results]
    
        if buff_size:
            logger.warning(f"PAYLOAD GUARDADO EN MEMORIA: '{sum(buff_size)} B', Y EN ARCHIVO DE SEGURIDAD")
            return names, buff_size

        else:
            logger.debug("NO SE ACTIVO MEMORIA DINÁMICA, SE REGRESAN LOS BYTES ESTIMADOS SOLAMENTE")
            
            return names, buff_size
    
    def process_images(self, query: List[str]) -> List[Tuple[str, str, int]]:
        """Procesa una sola imagen usando el método execute() uniforme de cada stager"""
        total_images = len(query)
        images = 0
        results: List[Tuple[str, str, int]] = []
        start_time = time.perf_counter()
        while query:
            images += 1
            image_data = query.pop(0)
            # Crear instancia fresca de DataFormatter para esta imagen
            manager = DataFormatter(self.logs_debug)
            # Crear contexto para esta ejecución
            image_name = get_ids(image_data, "file_name")
            context: Dict[str, Any] = {"image_data": image_data, "image_name": image_name}
            for i, stager in enumerate(self.all_stagers):
                stager_name = stager.__class__.__name__
                
                stager_init = time.perf_counter()
                manager = stager.execute(manager, context)

                if manager is None:
                    logger.error(f"'{image_name}' fallo en '{stager_name}', tiempo: {time.perf_counter() - stager_init:.6f}'s", exc_info=True)
                    if i != 1:
                        break
                    else:
                        continue
            
                if self.time_stages_log:
                    logger.warning(f"Fase de '{stager_name[:-6].upper()}' completada en: {time.perf_counter() - stager_init:.6f}'s")
            
            if manager is None:
                continue
            
            buff_size = manager.payload.buff_size if manager.payload else 0
            
            if self.memory:
                plain_text = manager.payload.payload if manager.payload else StringsModels.NO_MANGER.value
    
                if StringsModels.NO_MANGER.value == plain_text:
                    logger.info("PROCESO COMPLETADO, PRUEBA MOCK SE DEVUELVEN DATOS FALSOS")
                    continue
                    
                elif not write_temp_log(plain_text):
                    logger.error("NO SE PUDO GENERAR ARCHIVO DE SEGURIDAD")
                    continue
                else:
                    results.append((image_name, plain_text, buff_size))
            
            else:
                manager.reset_data()
                results.append((image_name, "", buff_size))
                
            logger.warning(f"Procesadas: {images} de '{total_images}' imagenes")
            
        total_processing_time = time.perf_counter() - start_time
        success_images = len(results)
        if 0 < success_images:
            logger.warning(f"'{success_images} de {total_images}' Archivos Digitalizados CORRECTAMENTE en: {total_processing_time}, promedio: {total_processing_time / total_images}'s por documento")
            return results
        else:
            logger.error(f"'{success_images} Archivos correctos en: {total_processing_time}, promedio: {total_processing_time / total_images}'s por documento")
            return []        
