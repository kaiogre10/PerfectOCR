# PerfectOCR/main_builder.py
import time
import logging
from typing import Optional, List, Dict, Any
from app.process_builder import ProcessingBuilder
from app.workflow_builder import WorkFlowBuilder
from core.pipeline.stagers_factory import StagersFactory
from core.domain.data_formatter import DataFormatter
from core.domain.models_manager import ModelsManager
from services.config_service import ConfigService
from services.cache_service import cleanup_project_cache

logger = logging.getLogger(__name__)

def activate_main(input_paths: List[str] | str, output_paths: List[str] | str, config_path: str, project_root: str) -> List[str]:
    
    try:
        # 1. Main activa al Configurador y valida parametros mínimos
        t0 = time.perf_counter()
        config_services = ConfigService(config_path)

        try:
            if config_services.validate_pipeline_config():
                logger.debug("Número mínimo de workers activos, se inicia el pipeline")
            else:
                return []
        except Exception as e:
            logger.error(f"No hay parametros mínimos para el pipeline, se detiene sin haber iniciado: {e}", exc_info=True)
            return []
        
        # 2. Main crea WorkFlowBuilder con configuración centralizada
        workflow_manager = WorkFlowBuilder(
            config_services=config_services,
            project_root=project_root,
            input_paths=input_paths,
            output_paths=output_paths
        )
        
        # 3. WorkflowManager analiza y reporta
        workflow_report = workflow_manager.count_and_plan()
        
        # 4. Iniciar modelos Singleton
        models_manager = ModelsManager.get_instance()
        models_config = config_services.models_config
        models_manager.initialize_models(models_config, project_root)

        # 5. CREAR STAGERS FACTORY UNA SOLA VEZ
        stagers_factory = StagersFactory(
            modules_config=config_services.modules_config,
            workers_order=config_services.workers_order,
            manager_config=config_services.manager_config,
            project_root=project_root
        )

        # 6. CREAR BUILDERS USANDO LA FACTORY
        builders = create_builders_with_factory(
            stagers_factory=stagers_factory,
            config_services=config_services,
            workflow_report=workflow_report,
            output_paths=output_paths
        )
        
        # 7. Main ejecuta procesamiento
        t4 = time.perf_counter()
        results = execute_processing(builders, workflow_report)
        logger.info(f"Procesamiento builder principal términado en {time.perf_counter()-t4:.6f}s")
        logger.info(f"Proceso términado completo en {time.perf_counter()-t0:.6f}s")
        return results
        
    except Exception as e:
        logger.fatal(f"Error fatal en main: {e}", exc_info=True)
        return []
        
    finally:
        try:
            cleanup_project_cache(project_root)
        except Exception as e:
            logging.error(f"Error durante la limpieza de caché: {e}", exc_info=True)

def create_builders_with_factory(
    stagers_factory: StagersFactory,
    config_services: ConfigService,
    workflow_report: Dict[str, Any],
    output_paths: List[str] | str,
) -> List[ProcessingBuilder]:
    """Crea builders usando StagersFactory centralizada."""
    
    builders: List[ProcessingBuilder] = []
    image_info_list = workflow_report.get('image_info', [])

    try:
        for image_data in image_info_list:
            # Contexto común para todos los stagers
            context: Dict[str, Any] = {
                "image_data": image_data,
            }

            input_stager = stagers_factory.create_image_prep_stager(context, output_paths)
            preprocessing_stager = stagers_factory.create_preprocessing_stager(context, output_paths)
            ocr_stager = stagers_factory.create_ocr_stager(context, output_paths)
            vectorization_stager = stagers_factory.create_vectorization_stager(context, output_paths)

            # Crear DataFormatter y ProcessingBuilder
            manager = DataFormatter()

            builder = ProcessingBuilder(
                config=config_services.manager_config,
                input_stager=input_stager,
                preprocessing_stager=preprocessing_stager,
                ocr_stager=ocr_stager,
                vectorization_stager=vectorization_stager,
                manager=manager
            )
            builders.append(builder)
        return builders

    except Exception as e:
        logging.error(f"Error fatal en create_builders: {e}", exc_info=True)
        return []

def execute_processing(builders: List['ProcessingBuilder'], workflow_report: Dict[str, Any]) -> Optional[List[str]]:
    """Ejecuta el procesamiento para cada builder."""
    db_paths: Dict[str, Any] = {}
    image_info_list = workflow_report.get('image_info', [])
    total_processing_time = 0.0
    builders_amount = len(builders)
    logger.info(f"Cantidad de Builder creados: {builders_amount}")
    try:
        for i, builder in enumerate(builders):
            if i < len(image_info_list):
                image_data = image_info_list[i]
                start_time = time.perf_counter()
                db_path = builder.process_single_image()
                end_time = time.perf_counter()
                total_processing_time += (end_time - start_time)
                db_paths[image_data.get('name', f'imagen_{i}')] = db_path

        if db_paths:
            promedio = total_processing_time / len(db_paths)
            logger.info(f"Total de imágenes: {len(db_paths)} en: {total_processing_time:.6f}s, promedio: {promedio:.6f}s")

        return ["db_path"]

    except Exception as e:
        logger.error(f"Error en el procesamiento: {e}", exc_info=True)
