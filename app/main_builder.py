# PerfectOCR/main_builder.py
from typing import Optional, List, Dict, Any, Tuple
from app.process_builder import ProcessingBuilder
from core.pipeline.stagers_factory import StagersFactory
from app.models_builder import ModelsBuilder
import services.system_service as system_service
from services.config_service import ConfigService
from core.utils.text_utils import format_elapsed_time
from app.conections_builder import ConectorsBuilder
import time
import logging

logger = logging.getLogger(__name__)

class MainBuilder:
    def __init__(self, config_service: ConfigService, project_root: str):
         self.project_root = project_root
         self.config_service = config_service

    def activate_main(self) -> List[str]:
        t0 = time.perf_counter()
        try:
            if not self.config_service or not self.project_root:
                system_service.cleanup_project_cache()
                logger.error("NO HAY RUTAS PRINCIPALES PARA PIPELINE, REVISAR MAIN\n"f"PROCESO DETENIDO: {time.perf_counter() - t0}s")
                return  []

            # 4. Iniciar modelos Singleton
            models_config = self.config_service.models_config
            if models_config:
                models_builder = ModelsBuilder.get_instance()
                if not models_builder.initialize_models(models_config, self.project_root):
                    logger.error("MODELOS NO SE PUDIERON INICIAR ABORTANDO")
                    system_service.cleanup_project_cache()
                    return []

            # Service analiza y reporta
            workflow_report = system_service.count_and_plan()
            if not workflow_report:
                logger.error(f"Error en rutas para imágenes, abortando proceso: {time.perf_counter() - t0:.6f}'s de ejecucusión", exc_info=True)
                system_service.cleanup_project_cache()
                return []

            if not self.config_service.no_modules:
                # CREAR STAGERS FACTORY UNA SOLA VEZ
                stagers_factory = StagersFactory(manager_config=self.config_service.manager_config, project_root=self.project_root)

                # CREAR UN ÚNICO BUILDER REUTILIZABLE
                processing_builder = self.create_single_builder(stagers_factory=stagers_factory, logs_config=self.config_service.logs_debug)
                if not processing_builder:
                    logger.error("No se pudo crear el ProcessingBuilder")
                    system_service.cleanup_project_cache()
                    return []

                final_payload_list = self.transform_image_to_df(processing_builder, workflow_report)

                conections_service = ConectorsBuilder(self.config_service.exporting_config)
                if conections_service.active_services and final_payload_list:
                    conections_service.set_up_connectors(final_payload_list)
                    # db_service = DataBaseService(dsn=None)
                    # if db_service.test_connection():
                    #     system_service.clean_db(db_service)
                    #     if not insert_data(db_service, final_payload_list):
                    #         logger.info(f"Tiempo en completar pipeline: {format_elapsed_time(time.perf_counter()-t0)}")

                system_service.cleanup_project_cache()
                logger.info(f"Tiempo en completar pipeline: {format_elapsed_time(time.perf_counter()-t0)}")
                return []

            logger.debug(f"Proceso debugger completo en {format_elapsed_time(time.perf_counter()-t0)}")
            system_service.cleanup_project_cache()
            return []

        except NameError as e:
            logger.error(f"ERROR FATAL EN BUILDERS, FINALIZANDO PROCESO: {e}", exc_info=True)
        return []

    def create_single_builder(self, stagers_factory: StagersFactory, logs_config: Dict[str, Any]) -> Optional[ProcessingBuilder]:
        """Crea un único builder reutilizable usando StagersFactory."""
        try:
            # Contexto inicial genérico (se enriquecerá en cada ejecución)
            context: Dict[str, Any] = {}

            input_stager = stagers_factory.create_image_prep_stager(context)
            preprocessing_stager = stagers_factory.create_preprocessing_stager(context)
            ocr_stager = stagers_factory.create_ocr_stager(context)
            vectorization_stager = stagers_factory.create_vectorization_stager(context)

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

    def transform_image_to_df(self, builder: ProcessingBuilder, workflow_report: Dict[str, Any]):
        """Ejecuta el procesamiento secuencial reutilizando el builder."""
        image_info_list: List[Dict[str, Any]] = workflow_report['image_info']
        total_images = len(image_info_list)
        logger.info(f"{total_images} IMAGENES PARA PROCESAR")

        succes_images: List[str] = []
        images_names = [names["name"][:-4] for names in image_info_list]
        final_results: List[Tuple[int, int]] = []
        start_time = time.perf_counter()

        for i, image_data in enumerate(image_info_list):
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
        total_success = len(failed_images)
        total_fails = total_images - total_success

        if total_success == total_images:
            logger.info(f"TODAS LAS IMÁGENES FUERON PROCESADAS CORRECTAMENTE EN {time_formatted}, promedio: {mean_process:.6f}'s")
        elif total_fails == total_images:
            logger.error(f"TODAS LAS IMÁGENES PRESENTARON FALLAS REVISAR CONFIGURACIÓN E IMÁGENES, TIEMPO: {time_formatted}")
        else:
            logger.info(f"'{total_success} / {total_images}' Archivos Digitalizados en: {time_formatted}, promedio: {mean_process:.6f}'s / documento")
            logger.info(f"IMAGENES EXITOSAS: {total_success}")
            logger.info(f"IMAGENES FALLADAS: {list(failed_images)}")

        logger.info(f"Direcciónes exitosas")
        return final_results