# PerfectOCR/main_builder.py
import time
from typing import Optional, List, Dict, Any, Tuple
from app.process_builder import ProcessingBuilder
from app.models_builder import ModelsBuilder
from core.pipeline.stagers_factory import StagersFactory
from services.config_service import ConfigService
import logging

time_mask = f"Tiempo: "
logger = logging.getLogger(__name__)

class MainBuilder:
    def __init__(self, config_service: ConfigService, project_root: str):
        self.project_root = project_root
        self.config_service = config_service

    def activate_main(self, workflow_report: List[Dict[str, Any]]) -> List[str]:
        t0 = time.perf_counter()
        try:
            if not workflow_report or not self.config_service:
                logger.info("NO HAY RUTAS PRINCIPALES PARA PIPELINE, REVISAR MAIN\n"f"PROCESO DETENIDO: {time.perf_counter() - t0}")
                return []

            # 4. Iniciar modelos Singleton
            models_config = self.config_service.models_config
            if models_config:
                models_builder = ModelsBuilder.get_instance()
                if not models_builder.initialize_models(models_config, self.project_root): # type: ignore
                    logger.info("MODELOS NO SE PUDIERON INICIAR ABORTANDO")
                    return []

            if not self.config_service.no_activate_modules:
                # CREAR UN ÚNICO BUILDER REUTILIZABLE
                processing_builder = self.create_single_builder(logs_config=self.config_service.logs_debug) # type: ignore
                if processing_builder is None:
                    return []

                self.transform_image_to_df(processing_builder, workflow_report)

                #if final_payload_list:
                 #   if postgre_local_service.start_postgres():
                  #      postgre_local_service.insert_payload(final_payload_list) # type: ignore

                logger.info(f"{time_mask}{time.perf_counter()-t0} total en completar el proceso")
                return []

            logger.warning(f"Proceso debugger completo, {time_mask}{time.perf_counter()-t0}")
            return []

        except Exception as e:
           logging.error(f"ERROR FATAL EN BUILDERS, FINALIZANDO PROCESO: {e}", exc_info=True)
        return []

    def create_single_builder(self, logs_config: Dict[str, Any]) -> Optional[ProcessingBuilder]:
        """Crea un único builder reutilizable usando StagersFactory."""
        try:
            # CREAR STAGERS FACTORY UNA SOLA VEZ
            manager_config = self.config_service.manager_config
            stagers_factory = StagersFactory(manager_config, project_root=self.project_root, stagging=self.config_service.create_stager) # type: ignore
            all_stagers = stagers_factory.get_all_stagers()

            # El manager se crea dentro del proceso de cada imagen, no aquí
            builder = ProcessingBuilder(all_stagers=all_stagers, logs_config=logs_config)
            return builder

        except AttributeError as e:
           logger.error(f"Error fatal en create_single_builder: {e}", exc_info=True)
        return None

    def transform_image_to_df(self, builder: ProcessingBuilder, workflow_report: List[Dict[str, Any]]):
        """Ejecuta el procesamiento secuencial reutilizando el builder."""
        # if builder is None:
        #     return
        
        total_images = len(workflow_report)
        logger.info(f"'{total_images}' IMAGENES PARA PROCESAR")

        succes_images: List[str] = []
        images_names = [names["name"][:-4] for names in workflow_report]
        final_results: List[Tuple[int, List[int]]] = []

        start_time = time.perf_counter()
        for i, image_data in enumerate(workflow_report):
            # Procesar imagen individualmente
            payload_dirs = builder.process_single_image(image_data)
            if payload_dirs is None:
                # logger.info(f"Fallo al procesar imagen: '{images_names[i]}'")
                continue
            else:
                image_name = images_names[i]
                succes_images.append(image_name)
                final_results.append(payload_dirs)
                logger.info(f"IMAGEN '{image_name}' # {(i + 1)} de '{total_images}' imágenes")
                continue

        total_processing_time = time.perf_counter() - start_time
        del builder
        mean_process = f"{(total_processing_time / total_images):.6f}"

        failed_images = set(images_names).difference(set(succes_images))
        total_fails = len(failed_images)
        
        if total_fails == 0:
            logger.info(f"TODAS LAS IMÁGENES FUERON PROCESADAS CORRECTAMENTE EN {time_mask}{total_processing_time}, promedio: {mean_process}'s")
        elif total_fails == total_images:
            logger.info(f"TODAS LAS IMÁGENES PRESENTARON FALLAS REVISAR CONFIGURACIÓN E IMÁGENES, {time_mask}{total_processing_time}, promedio: {mean_process}")
        else:
            logger.info(f"'{total_images - total_fails} de {total_images}' Archivos Digitalizados en: {time_mask}{total_processing_time}, promedio: {mean_process}'s / documento")

            logger.info(f"IMAGENES EXITOSAS: {succes_images}")
            logger.info(f"IMAGENES FALLADAS: {list(failed_images)}")

        return final_results