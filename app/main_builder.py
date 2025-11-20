# PerfectOCR/main_builder.py
import time
from typing import Optional, List, Dict, Any
from app.process_builder import ProcessingBuilder
from app.workflow_builder import WorkFlowBuilder
from core.pipeline.stagers_factory import StagersFactory
from core.domain.data_formatter import DataFormatter
from services.config_service import ConfigService
from core.domain.models_manager import ModelsManager
import logging

logger = logging.getLogger(__name__)

def activate_main(input_paths: List[str] | str, output_paths: List[str] | str, config_path: str, project_root: str) -> List[str]:
    
    try:
        # 1. Main activa al Configurador y valida parametros mínimos
        t0 = time.perf_counter()
        config_services = ConfigService(config_path)

        if config_services.validate_pipeline_config():
            logger.debug("Número mínimo de workers activos, se inicia el pipeline")
        else:
            logger.error(f"Proceso terminado en: {time.perf_counter()-t0:.6f}s debido a configuración insuficiente")
            return []
        
        # 2. Main crea WorkFlowBuilder con configuración centralizada
        workflow_manager = WorkFlowBuilder(
            config=config_services.processing_config,
            project_root=project_root,
            input_paths=input_paths,
        )
        
        # 3. WorkflowManager analiza y reporta
        workflow_report = workflow_manager.count_and_plan()
        
        # 4. Iniciar modelos Singleton
        models_manager = ModelsManager.get_instance()
        models_manager.initialize_models(config_services.models_config)

        # 5. CREAR STAGERS FACTORY UNA SOLA VEZ
        stagers_factory = StagersFactory(
            manager_config=config_services.manager_config,
            project_root=project_root
        )

        # 6. CREAR BUILDERS USANDO LA FACTORY
        stagers = config_services.create_stager()
        builders = create_builders_with_factory(
            stagers = stagers,
            stagers_factory=stagers_factory,
            workflow_report=workflow_report,
            output_paths=output_paths
        )
        
        # 7. Main ejecuta procesamiento
        t4 = time.perf_counter()
        results = execute_processing(builders, workflow_report)
        logger.info(f"Procesamiento builder principal términado en {time.perf_counter()-t4:.6f}s")
        logger.info(f"Proceso términado completo en {time.perf_counter()-t0:.6f}s")
        return results #type: ignore
        
    except Exception as e:
        logger.fatal(f"Error fatal en main: {e}", exc_info=True)
        return []
        
    finally:
        try:
            from services.cache_service import cleanup_project_cache
            cleanup_project_cache(project_root)
        except Exception as e:
            logging.error(f"Error durante la limpieza de caché: {e}", exc_info=True)

def create_builders_with_factory(stagers: List[str], stagers_factory: StagersFactory, workflow_report: Dict[str, Any], output_paths: List[str] | str) -> List[ProcessingBuilder]:
    """Crea builders usando StagersFactory centralizada."""
    builders: List[ProcessingBuilder] = []
    image_info_list = workflow_report.get('image_info', {}) 

    try:
        for image_data in image_info_list:
            # Contexto común para todos los stagers
            context: Dict[str, Any] = {
                "image_data": image_data,
            }
            
            stagers_created = 0
            if "imagepre_stage" in stagers:
                input_stager = stagers_factory.create_image_prep_stager(context, output_paths)
                stagers_created += 1

            if "preprocessing_stage" in stagers:
                preprocessing_stager = stagers_factory.create_preprocessing_stager(context, output_paths)
                stagers_created += 1

            if "ocr_stage" in stagers:
                ocr_stager = stagers_factory.create_ocr_stager(context, output_paths)
                stagers_created += 1

            if "vector_stage" in stagers:
                vectorization_stager = stagers_factory.create_vectorization_stager(context, output_paths)

                stagers_created += 1

            logger.info(f"STAGER CREADOS: {stagers_created}")

            # Crear DataFormatter y ProcessingBuilder
            manager = DataFormatter()

            builder = ProcessingBuilder(
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
    total_img = 0
    logger.info(f"Cantidad de Builder creados: {builders_amount}")

    try:
        for i, builder in enumerate(builders):
            if i < len(image_info_list):
                image_data = image_info_list[i]
                start_time = time.perf_counter()
                db_path = builder.process_single_image()
                total_img +=1
                image_processing_time = time.perf_counter() - start_time
                total_processing_time += image_processing_time
                db_paths[image_data.get('name', f'imagen_{i}')] = db_path
                logger.warning(f"IMAGEN '{image_data.get('name')}' {total_img} de {builders_amount} PROCESADA EN: {image_processing_time:.6f}s")

        if db_paths:
            mean_time = total_processing_time / len(db_paths)
            logger.warning(f"Total de imágenes: {len(db_paths)} en: {total_processing_time:.6f}s, promedio: {mean_time:.6f}s")

        return ["db_path"]

    except Exception as e:
        logger.error(f"Error en el procesamiento: {e}", exc_info=True)