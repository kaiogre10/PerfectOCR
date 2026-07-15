# PerfectOCR/main_builder.py
import time
from typing import Optional, List, Tuple
from app.process_builder import ProcessingBuilder
from app.models_builder import ModelsBuilder
from core.factory.main_factory import MainFactory
from services.config_service import ConfigService
import logging
import bitmath # type: ignore

time_mask = f"Tiempo: "
logger = logging.getLogger(__name__)

class MainBuilder:
    __slots__ = ("project_root", "config_service")
    def __init__(self, config_service: ConfigService, project_root: str):
        self.project_root = project_root
        self.config_service = config_service

    def activate_main(self, workflow_report: List[str]) -> List[str]:
        t0 = time.perf_counter()
        try:
            if not workflow_report or not self.config_service:
                logger.info("NO HAY RUTAS PRINCIPALES PARA PIPELINE, REVISAR MAIN\n"f"PROCESO DETENIDO: {time.perf_counter() - t0}")
                return []
                    # CREAR UN ÚNICO BUILDER REUTILIZABLE
            processing_builder = self.create_single_builder()  # type: ignore
            if processing_builder is None:
                return []

            models_config = self.config_service.models_config
            if models_config:
                models_builder = ModelsBuilder.get_instance()
                if not models_builder.initialize_models(models_config):
                    logger.error("MODELOS NO SE PUDIERON INICIAR: ABORTANDO")
                    return []
                
                elif models_config.get("update_model"):
                    logger.info("ACTUALIZAICIÓN PARA WORD FINDER, SALIENDO")
                    return []
                
            if not self.config_service.no_activate_modules:
                self.transform_image_to_df(processing_builder, workflow_report)

                #if final_payload_list:
                 #   if postgre_local_service.start_postgres():
                  #      postgre_local_service.insert_payload(final_payload_list) # type: ignore
                logger.warning(f"{time_mask}{time.perf_counter()-t0} total en completar el proceso de tranformación de imagenes")
                return []

            logger.warning(f"No modules completo, {time_mask}{time.perf_counter()-t0}")
            return []

        except Exception as e:
           logging.info(f"ERROR FATAL EN BUILDERS, FINALIZANDO PROCESO: {e}", exc_info=True)
        return []

    def create_single_builder(self) -> Optional[ProcessingBuilder]:
        """Crea un único builder reutilizable usando StagersFactory."""
        try:
            stagers_factory = MainFactory(self.project_root, self.config_service.stagers_config)
            all_stagers = stagers_factory.get_all_stagers(stagging=self.config_service.create_stager)
            if not all_stagers:
                raise AttributeError("NO SE CREÓ LA FACTORY")
            # El manager se crea dentro del proceso de cada imagen, no aquí
            builder = ProcessingBuilder(self.project_root, all_stagers=all_stagers, logs_debug=self.config_service.logs_debug)
            return builder

        except AttributeError as e:
           logger.error(f"Error fatal en create_single_builder: {e}", exc_info=True)
        return None

    def transform_image_to_df(self, builder: ProcessingBuilder, workflow_report: List[str]):
        """Ejecuta el procesamiento secuencial reutilizando el builder."""
        total_images = len(workflow_report)
        start_time = time.perf_counter()
        payloads_list = builder.process_query(workflow_report)
        total_processing_time = time.perf_counter() - start_time
        del builder
        
        if payloads_list is None:
            return []
        else:
            success_images = [name for name in payloads_list[0]]
            payload_sizes = [bitmath.Byte(sizes) for sizes in payloads_list[1]]
            mean_process = f"{(total_processing_time / total_images):.6f}"
            failed_images = {names.replace("\\", "/").split("/")[-1].split(".")[0] for names in workflow_report}.difference(set(success_images))
            total_fails = len(failed_images)
            
            if total_fails == 0:
                logger.warning(f"TODAS LAS IMÁGENES FUERON PROCESADAS CORRECTAMENTE EN {time_mask}{total_processing_time}, promedio: {mean_process}'s")
    
            elif total_fails == total_images:
                logger.warning(f"TODAS LAS IMÁGENES PRESENTARON FALLAS REVISAR CONFIGURACIÓN E IMÁGENES, {time_mask}{total_processing_time}, promedio: {mean_process}")
    
            else:
                logger.debug(f"IMAGENES EXITOSAS:\n"f"{success_images}\n"f"----------------------------")
                logger.debug(f"IMAGENES FALLIDAS:\n"f"{failed_images}\n"f"----------------------------")
        
            logger.info(f"SIZE/SENDED: {sum(payload_sizes)} / {len(success_images)}")
            return payloads_list

    def send_payload_pack(self, size: bitmath.Any, total_payloads: int) -> Tuple[bitmath.Any, int]:
        payload_resized = bitmath.best_prefix(size)
        logger.info(f"TAMAÑO DEL PAYLOAD: '{payload_resized}', enviados: '{total_payloads}'")
        return (payload_resized, total_payloads)