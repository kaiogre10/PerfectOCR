# PerfectOCR/main_builder.py
import logging
import time
from typing import Optional, List, Dict, Any, Tuple
from app.process_builder import ProcessingBuilder
from app.models_builder import ModelsBuilder
from core.pipeline.stagers_factory import StagersFactory
from services.config_service import ConfigService
#import services.postgre_local_service as postgre_local_service
from core.utils.text_utils import format_elapsed_time

logger = logging.getLogger(__name__)

class MainBuilder:
    def __init__(self, config_service: ConfigService, project_root: str):
        self.project_root = project_root
        self.config_service = config_service

    def activate_main(self, workflow_report: List[Dict[str, Any]]) -> List[str]:
        t0 = time.perf_counter()
        try:
            if not workflow_report or not self.config_service:
                logger.error("NO HAY RUTAS PRINCIPALES PARA PIPELINE, REVISAR MAIN\n"f"PROCESO DETENIDO: {time.perf_counter() - t0}s")
                return  []

            # 4. Iniciar modelos Singleton
            models_config = self.config_service.models_config
            if models_config:
                models_builder = ModelsBuilder.get_instance()
                if not models_builder.initialize_models(models_config, self.project_root): # type: ignore
                    logger.error("MODELOS NO SE PUDIERON INICIAR ABORTANDO")
                    return []

            if not self.config_service.no_activate_modules:
                # CREAR UN ÚNICO BUILDER REUTILIZABLE
                processing_builder = self.create_single_builder(logs_config=self.config_service.logs_debug) # type: ignore
                if not processing_builder:
                    logger.error("No se pudo crear el ProcessingBuilder")
                    
                    return []

                self.transform_image_to_df(processing_builder, workflow_report)

                #if final_payload_list:
                 #   if postgre_local_service.start_postgres():
                  #      postgre_local_service.insert_payload(final_payload_list) # type: ignore

                logger.info(f"Tiempo en completar pipeline: {format_elapsed_time(time.perf_counter()-t0)}")
                return []

            logger.info(f"Proceso debugger completo en {format_elapsed_time(time.perf_counter()-t0)}")
            return []

        except NameError as e:
            logger.error(f"ERROR FATAL EN BUILDERS, FINALIZANDO PROCESO: {e}", exc_info=True)
        return []

    def create_single_builder(self, logs_config: Dict[str, Any]) -> Optional[ProcessingBuilder]:
        """Crea un único builder reutilizable usando StagersFactory."""
        try:
            # CREAR STAGERS FACTORY UNA SOLA VEZ
            manager_config = self.config_service.manager_config
            stagers_factory = StagersFactory(manager_config, project_root=self.project_root) # type: ignore
            input_stager = stagers_factory.create_image_prep_stager()
            preprocessing_stager = stagers_factory.create_preprocessing_stager()
            ocr_stager = stagers_factory.create_ocr_stager()
            vectorization_stager = stagers_factory.create_vectorization_stager()

            # El manager se crea dentro del proceso de cada imagen, no aquí
            builder = ProcessingBuilder(
                input_stager=input_stager,
                preprocessing_stager=preprocessing_stager,
                ocr_stager=ocr_stager,
                vectorization_stager=vectorization_stager,
                logs_config=logs_config
            )
            return builder

        except Exception as e:
            logger.error(f"Error fatal en create_single_builder: {e}", exc_info=True)
        return None        

    def transform_image_to_df(self, builder: ProcessingBuilder, workflow_report: List[Dict[str, Any]]):
        """Ejecuta el procesamiento secuencial reutilizando el builder."""
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
                logger.error(f"Fallo al procesar imagen: '{images_names[i]}'")
                continue
            else:
                image_name = images_names[i]
                succes_images.append(image_name)
                final_results.append(payload_dirs)
                logger.info(f"IMAGEN '{image_name}', '# {(i + 1)}' de '{total_images}' imágenes")
                continue

        total_processing_time = time.perf_counter() - start_time
        mean_process = (total_processing_time / total_images)
        time_formatted = format_elapsed_time(total_processing_time)

        failed_images = set(images_names).difference(set(succes_images))
        total_fails = len(failed_images)
        
        if total_fails == 0:
            logger.info(f"TODAS LAS IMÁGENES FUERON PROCESADAS CORRECTAMENTE EN {time_formatted}, promedio: {mean_process:.6f}'s")
        elif total_fails == total_images:
            logger.error(f"TODAS LAS IMÁGENES PRESENTARON FALLAS REVISAR CONFIGURACIÓN E IMÁGENES, TIEMPO: {time_formatted}")
        else:
            logger.info(f"'{total_images - total_fails} de {total_images}' Archivos Digitalizados en: {time_formatted}, promedio: {mean_process:.6f}'s / documento")

            logger.info(f"IMAGENES EXITOSAS: {succes_images}")
            logger.info(f"IMAGENES FALLADAS: {list(failed_images)}")

        return final_results