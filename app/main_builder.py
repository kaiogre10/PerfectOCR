# PerfectOCR/main_builder.py
from typing import Optional, List, Dict, Any
from app.process_builder import ProcessingBuilder
from app.workflow_builder import WorkFlowBuilder
from core.pipeline.stagers_factory import StagersFactory
from core.domain.data_formatter import DataFormatter
from services.config_service import ConfigService
from core.domain.models_manager import ModelsManager
from services.cache_service import cleanup_project_cache
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
        config_services = ConfigService(config_path, TEST_MODE)
        
        # 2. Main crea WorkFlowBuilder con configuración centralizada
        workflow_manager = WorkFlowBuilder(builder_config=config_services.processing_config, project_root=project_root, input_paths=input_paths)
        
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
        
        # 5. CREAR STAGERS FACTORY UNA SOLA VEZ
        stagers_factory = StagersFactory(manager_config=config_services.manager_config, project_root=project_root)

        # 6. CREAR BUILDERS USANDO LA FACTORY
        builders = create_builders_with_factory(stagers_factory=stagers_factory, workflow_report=workflow_report, output_paths=output_paths)
        
        # 7. Main ejecuta procesamiento
        t4 = time.perf_counter()
        results = execute_processing(builders, workflow_report)
        logger.debug(f"Procesamiento builder principal términado en {time.perf_counter()-t4:.6f}s")
        logger.debug(f"Proceso términado completo en {time.perf_counter()-t0:.6f}s")

        cleanup_project_cache(project_root)
        return results #type: ignore
        
    except NameError as e:
        logger.error(f"ERROR FATAL EN BUILDERS, FINALIZANDO PROCESO: {e}", exc_info=True)
    return []
    
def create_builders_with_factory(stagers_factory: StagersFactory, workflow_report: Dict[str, Any], output_paths: List[str]) -> List[ProcessingBuilder]:
    """Crea builders usando StagersFactory centralizada."""
    builders: List[ProcessingBuilder] = []
    image_info_list = workflow_report.get('image_info', {}) 

    try:
        for image_data in image_info_list:
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
                input_stager=input_stager,
                preprocessing_stager=preprocessing_stager,
                ocr_stager=ocr_stager,
                vectorization_stager=vectorization_stager,
                manager=manager
            )
            builders.append(builder)

        return builders

    except Exception as e:
        logger.error(f"Error fatal en create_builders: {e}", exc_info=True)
    return []

def execute_processing(builders: List['ProcessingBuilder'], workflow_report: Dict[str, Any]) -> Optional[List[str]]:
    """Ejecuta el procesamiento para cada builder."""
    db_paths: Dict[str, Any] = {}
    image_info_list = workflow_report['image_info']
    total_processing_time = 0.0
    builders_amount = len(builders)
    total_img = 0
    logger.debug(f"Cantidad de Builder creados: {builders_amount}")

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
                logger.warning(f"IMAGEN '{image_data.get('name')}', #{total_img} de un total de {builders_amount} imágenes. PROCESADA EN: {image_processing_time:.6f}s")

        if db_paths:
            mean_time = total_processing_time / len(db_paths)
            logger.warning(f"Total de imágenes: '{len(db_paths)}' en: {total_processing_time:.6f}s, promedio: {mean_time:.6f}s")

        return ["db_path"]

    except Exception as e:
        logger.error(f"Error en el procesamiento: {e}", exc_info=True)