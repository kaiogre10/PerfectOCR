# PerfectOCR/main_builder.py
from typing import Optional, List, Dict, Any, Tuple
from app.process_builder import ProcessingBuilder
from core.pipeline.stagers_factory import StagersFactory
from services.config_service import ConfigService
from app.models_manager import ModelsManager
from services.db_service import DataBaseService
import services.system_service as system_service
from core.utils.text_utils import format_elapsed_time
import time
import pandas as pd # type: ignore
import logging

PROJECT_ROOT = ""

def set_project_root(project_root: str):
    global PROJECT_ROOT
    PROJECT_ROOT = project_root # type: ignore

logger = logging.getLogger(__name__)

def activate_main(output_paths: List[str], config_path: str, TEST_MODE: bool) -> List[str]:
    t0 = time.perf_counter()
    try:
        if not config_path or not PROJECT_ROOT:
            system_service.cleanup_project_cache()
            logger.error("NO HAY RUTAS PRINCIPALES PARA PIPELINE, REVISAR MAIN\n"f"PROCESO DETENIDO: {time.perf_counter() - t0}s")
            return  []
        
        # 1. Main activa al Configurador y valida parametros mínimos
        config_services = ConfigService(config_path, TEST_MODE, output_paths)
        
        # 3. Service analiza y reporta
        workflow_report = system_service.count_and_plan(config=config_services.system_config)
        if not workflow_report:
            logger.error(f"Error en rutas para imágenes, abortando proceso: {time.perf_counter() - t0:.6f}'s de ejecucusión", exc_info=True)
            system_service.cleanup_project_cache()
            return []
        
        # 4. Iniciar modelos Singleton
        models_config = config_services.models_config
        if models_config:
            models_manager = ModelsManager.get_instance()
            if not models_manager.initialize_models(models_config, PROJECT_ROOT):
                logger.info("MODELOS NO SE PUDIERON INICIAR ABORTANDO")
                system_service.cleanup_project_cache()
                return []
        
        if not config_services.no_modules:
            # 5. CREAR STAGERS FACTORY UNA SOLA VEZ
            stagers_factory = StagersFactory(manager_config=config_services.manager_config, project_root=PROJECT_ROOT)
            
            # 6. CREAR UN ÚNICO BUILDER REUTILIZABLE
            logs_config = config_services.logs_debug
            processing_builder = create_single_builder(stagers_factory=stagers_factory, logs_config=logs_config)
            if not processing_builder:
                logger.error("No se pudo crear el ProcessingBuilder")
                system_service.cleanup_project_cache()
                return []
        
            # 7. Main ejecuta procesamiento secuencial usando el builder único
            # t4 = time.perf_counter()
            final_df_list = transform_image_to_df(processing_builder, workflow_report)
            # logger.info(f"Procesamiento builder principal términado en {time.perf_counter()-t4:.6f}s")
            if final_df_list and config_services.db_config:
                db_service = DataBaseService(dsn=None)
                if db_service.test_connection():
                    system_service.clean_db(db_service)
                    if not insert_data(db_service, final_df_list):
                        logger.info(f"Tiempo en completar pipeline: {format_elapsed_time(time.perf_counter()-t0)}")

            system_service.cleanup_project_cache()
            logger.info(f"Tiempo en completar pipeline: {format_elapsed_time(time.perf_counter()-t0)}")
            return []

        logger.debug(f"Proceso debugger completo en {format_elapsed_time(time.perf_counter()-t0)}")
        system_service.cleanup_project_cache()
        return []
        
    except NameError as e:
        logger.error(f"ERROR FATAL EN BUILDERS, FINALIZANDO PROCESO: {e}", exc_info=True)
    return []
    
def create_single_builder(stagers_factory: StagersFactory, logs_config: Dict[str, Any]) -> Optional[ProcessingBuilder]:
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

def transform_image_to_df(builder: ProcessingBuilder, workflow_report: Dict[str, Any]) -> List[Tuple[pd.DataFrame, Dict[str, Any]]]:
    """Ejecuta el procesamiento secuencial reutilizando el builder."""
    image_info_list: List[Dict[str, Any]] = workflow_report['image_info']
    total_images = len(image_info_list)
    logger.info(f"{total_images} IMAGENES PARA PROCESAR")

    final_results: List[Tuple[pd.DataFrame, Dict[str, Any]]] = []
    failed_images: List[str] = []
    images_names = [names["name"][:-4] for names in image_info_list]
    start_time = time.perf_counter()

    for i, image_data in enumerate(image_info_list):
        # Procesar imagen individualmente
        final_df = builder.process_single_image(image_data)
        image_name = images_names[i]
        if final_df is None:
            logger.error(f"Fallo al procesar imagen: '{image_name}'")
            failed_images.append(image_name)
            continue
        else:
            final_results.append(final_df)
            logger.info(f"IMAGEN '{image_name}', '# {(i + 1)}' de '{total_images}' imágenes")
            continue

    total_processing_time = time.perf_counter() - start_time
    mean_process = (total_processing_time / total_images)
    time_formatted = format_elapsed_time(total_processing_time)

    succcess_image = set(images_names).difference(set(failed_images))
    total_fails = len(failed_images)
    total_success = total_images - total_fails

    if total_success == total_images:
        logger.info(f"TODAS LAS IMÁGENES FUERON PROCESADAS CORRECTAMENTE EN {time_formatted}, promedio: {mean_process:.6f}'s")
    elif total_fails == total_images:
        logger.error(f"TODAS LAS IMÁGENES PRESENTARON FALLAS REVISAR CONFIGURACIÓN E IMÁGENES, TIEMPO: {time_formatted}")
    else:
        logger.info(f"'{total_success} / {total_images}' Archivos Digitalizados en: {time_formatted}, promedio: {mean_process:.6f}'s / documento")
        logger.info(f"IMAGENES EXITOSAS: {list(succcess_image)}")
        logger.info(f"IMAGENES FALLADAS: {failed_images}")
        
    return final_results
    
def insert_data(db_service: DataBaseService, final_df_list: List[Tuple[pd.DataFrame, Dict[str, Any]]]):
    return db_service.insert_payload(final_df_list)