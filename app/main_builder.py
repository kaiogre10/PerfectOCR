# PerfectOCR/main_builder.py
from typing import Optional, List, Dict, Any
from app.process_builder import ProcessingBuilder
from app.workflow_builder import WorkFlowBuilder
from core.pipeline.stagers_factory import StagersFactory
from services.config_service import ConfigService
from core.domain.models_manager import ModelsManager
from services.cache_service import cleanup_project_cache
from datetime import datetime
import time
import logging

logger = logging.getLogger(__name__)

def activate_main(input_paths: List[str], output_paths: List[str], config_path: str, project_root: str, TEST_MODE: bool) -> List[str]:
    t0 = time.perf_counter()
    try:
        if not input_paths or not config_path or not project_root:
            import os
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cleanup_project_cache(project_root)
            logger.error("NO HAY RUTAS PRINCIPALES PARA PIPELINE, REVISAR MAIN"
                         "\n"f"PROCESO DETENIDO: {time.perf_counter() - t0}s")
            return  []
        
        # 1. Main activa al Configurador y valida parametros mínimos
        # t1 = time.perf_counter()
        config_services = ConfigService(config_path, TEST_MODE, output_paths)
        # logger.info(f"Config service completo en {time.perf_counter()-t1:.6f}s")
        
        # 2. Main crea WorkFlowBuilder con configuración centralizada
        workflow_manager = WorkFlowBuilder(builder_config=config_services.utils_config, project_root=project_root, input_paths=input_paths)
        
        # 3. WorkflowManager analiza y reporta
        workflow_report = workflow_manager.count_and_plan()
        if not workflow_report:
            logger.error(f"Error en rutas para imágenes, abortando proceso:", exc_info=True)
            cleanup_project_cache(project_root)
            return []
        
        # 4. Iniciar modelos Singleton
        models_config = config_services.models_config
        if models_config:
            models_manager = ModelsManager.get_instance()
            models_manager.initialize_models(models_config)
        
        if not config_services.no_modules:
        # 5. CREAR STAGERS FACTORY UNA SOLA VEZ
            stagers_factory = StagersFactory(manager_config=config_services.manager_config, project_root=project_root)
            
            # 6. CREAR UN ÚNICO BUILDER REUTILIZABLE
            processing_builder = create_single_builder(stagers_factory=stagers_factory, output_paths=output_paths)
            if not processing_builder:
                logger.error("No se pudo crear el ProcessingBuilder")
                cleanup_project_cache(project_root)
                return []
        
            # 7. Main ejecuta procesamiento secuencial usando el builder único
            # t4 = time.perf_counter()
            results = execute_sequential_processing(processing_builder, workflow_report)
            # logger.info(f"Procesamiento builder principal términado en {time.perf_counter()-t4:.6f}s")
            cleanup_project_cache(project_root)
            return results if results is not None else []
            
        logger.debug(f"Proceso debugger completo en {time.perf_counter()-t0:.6f}s")
        cleanup_project_cache(project_root)
        return []
        
    except NameError as e:
        logger.error(f"ERROR FATAL EN BUILDERS, FINALIZANDO PROCESO: {e}", exc_info=True)
    return []
    
def create_single_builder(stagers_factory: StagersFactory, output_paths: List[str]) -> Optional[ProcessingBuilder]:
    """Crea un único builder reutilizable usando StagersFactory."""
    try:
        # Contexto inicial genérico (se enriquecerá en cada ejecución)
        context: Dict[str, Any] = {}
        
        input_stager = stagers_factory.create_image_prep_stager(context, output_paths)
        preprocessing_stager = stagers_factory.create_preprocessing_stager(context, output_paths)
        ocr_stager = stagers_factory.create_ocr_stager(context, output_paths)
        vectorization_stager = stagers_factory.create_vectorization_stager(context, output_paths)
        
        # El manager se crea dentro del proceso de cada imagen, no aquí
        builder = ProcessingBuilder(
            input_stager=input_stager,
            preprocessing_stager=preprocessing_stager,
            ocr_stager=ocr_stager,
            vectorization_stager=vectorization_stager
        )
        return builder

    except Exception as e:
        logger.error(f"Error fatal en create_single_builder: {e}", exc_info=True)
        return None

def execute_sequential_processing(builder: ProcessingBuilder, workflow_report: Dict[str, Any]) -> Optional[List[str]]:
    """Ejecuta el procesamiento secuencial reutilizando el builder."""
    db_paths: Dict[str, Any] = {}
    image_info_list = workflow_report['image_info']
    total_processing_time = 0.0
    total_images = len(image_info_list)
    processed_count = 0
    
    logger.debug(f"Iniciando procesamiento secuencial para {total_images} imágenes.")

    try:
        for i, image_data in enumerate(image_info_list):
            start_time = time.perf_counter()
            image_name = image_data.get('name', f'imagen_{i}')
            
            # Procesar imagen individualmente
            manager_result = builder.process_single_image(image_data)
            
            # Simulación de extracción de resultados
            db_path = "db_path" 
            
            processed_count += 1
            image_processing_time = time.perf_counter() - start_time
            total_processing_time += image_processing_time
            
            if manager_result:
                db_paths[image_name] = db_path
                logger.debug(f"IMAGEN '{image_name}', #{processed_count} de {total_images}. PROCESADA EN: {image_processing_time:.6f}s")
            else:
                logger.error(f"Fallo al procesar imagen: {image_name}")

        if db_paths:
            mean_time = total_processing_time / len(db_paths)
            logger.warning(f"'{len(db_paths)}' Archivos Digitalizados el '{datetime.now().strftime('%m/%d %H:%M:%S')}' en: {total_processing_time:.6f}s, promedio: {mean_time:.6f}s")

        return ["db_path"]

    except Exception as e:
        logger.error(f"Error en el procesamiento secuencial: {e}", exc_info=True)
        return []
