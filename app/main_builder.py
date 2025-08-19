# PerfectOCR/activate_main.py
import time
import logging
from typing import Optional, List, Dict, Any
from app.process_builder import ProcessingBuilder
from app.workflow_builder import WorkFlowBuilder
from core.pipeline.input_stager import InputStager
from core.pipeline.preprocessing_stager import PreprocessingStager
from core.pipeline.ocr_stager import OCRStager
from core.pipeline.vectorization_stager import VectorizationStager
from core.factory.main_factory import MainFactory
from core.workers.image_preparation.image_loader import ImageLoader
from core.domain.data_formatter import DataFormatter
from core.domain.ocr_motor_manager import PaddleManager
from services.config_service import ConfigService
from services.cache_service import cleanup_project_cache

logger = logging.getLogger(__name__)

def activate_main(input_paths: Optional[List[str]], output_paths: Optional[List[str]], config_path: str, project_root: str) -> Dict[str, Any]:
    
    try:
        # 1. Main activa al Configurador
        t0 = time.perf_counter()
        # logging.debug("Activando ConfigManager con validación robusta...")
        config_services = ConfigService(config_path)
        # logging.info(f"ConfigManager iniciado en {time.perf_counter()-t0:.6f}s")
        
        # 2. Main crea WorkFlowBuilder con configuración centralizada
        # t1 = time.perf_counter()
        # logging.debug("Creando WorkFlowBuilder...")
        workflow_manager = WorkFlowBuilder(
            config_services=config_services,
            project_root=project_root,
            input_paths=input_paths,
        )
        # logging.info(f"WorkFlowBuilder completado en {time.perf_counter()-t1:.6f}s")
        
        # 3. WorkflowManager analiza y reporta
        # t2 = time.perf_counter()
        # logging.debug("Analizando imágenes disponibles...")
        workflow_report = workflow_manager.count_and_plan()
        # logging.info(f"Analisis Wrokflow builder completado en {time.perf_counter()-t2:.6f}s")
        
        # 4. Iniciar Paddle Singleton
        # t21 = time.perf_counter()
        paddle_manager = PaddleManager.get_instance()
        paddle_config = config_services.paddle_config
        paddle_manager.initialize_engines(paddle_config)
        # logging.info(f"PaddleManager iniciado en {time.perf_counter()-t21:.6f}s")

        # 5. Main crea builders según el reporte
        # t3 = time.perf_counter()
        # logging.debug("Creando builders según análisis")
        builders = create_builders(config_services, project_root, workflow_report, output_paths)
        # logging.info(f"Builders creados en {time.perf_counter()-t3:.6f}s")
        
        # 6. Main ejecuta procesamiento
        t4 = time.perf_counter()
        # logging.debug("Iniciando procesamiento...")
        results = execute_processing(builders, workflow_report)
        logging.info(f"Procesamiento builder principal términado en {time.perf_counter()-t4:.6f}s")
        logging.info(f"Proceso términado completo en {time.perf_counter()-t0:.6f}s")
        return results
        
    except Exception as e:
        logging.error(f"Error fatal en main: {e}", exc_info=True)
        return {"error": str(e)}
        
    finally:
        try:
            cleanup_project_cache(project_root)
        except Exception as cleanup_error:
            logging.error(f"Error durante la limpieza de caché: {cleanup_error}", exc_info=True)

def create_builders(config_services: ConfigService, project_root: str, workflow_report: Dict[str, Any], output_paths: Optional[List[str]])-> List[ProcessingBuilder]:
    """Crea builders para cada imagen encontrada usando inyecciones en cascada."""
    context = {}
    builders: List[ProcessingBuilder] = []
    image_info_list = workflow_report.get('image_info', [])
        
    for image_data in image_info_list:
        worker_factory = MainFactory(
            config_services.modules_config,
            project_root=project_root
        )
        
        manager = DataFormatter()
        geometry_detector = config_services.paddle_det_config
        paddle_wrapper = config_services.paddle_rec_config

        context = {
           "geometry_detector": geometry_detector,
            "paddle_wrapper": paddle_wrapper
        }

        image_load_factory = worker_factory.get_image_preparation_factory()
        image_prep_workers = image_load_factory.create_workers([
            "cleaner", "angle_corrector", "geometry_detector", "polygon_extractor"
        ], context)

        preprocessing_factory = worker_factory.get_preprocessing_factory()
        preprocessing_workers = preprocessing_factory.create_workers(
            ["moire", "sp", "gauss", "clahe", "sharp", "binarization", "fragmentator"],
            context
        )
        
        ocr_factory = worker_factory.get_ocr_factory()
        ocr_workers = ocr_factory.create_workers(["paddle_wrapper", "text_cleaner"], context)

        vectorizing_factory = worker_factory.get_vectorizing_factory()
        vectorization_workers = vectorizing_factory.create_workers(
            ["lineal", "dbscan", "table_structurer", "math_max"], 
            context
        )
        
        image_loader = ImageLoader(
            image_info=image_data,
            project_root=project_root,
        )
        
        input_stager = InputStager(
            workers=image_prep_workers,
            image_loader = image_loader,
            project_root=project_root
        )
            
        preprocessing_stager = PreprocessingStager(
            workers=preprocessing_workers,
            stage_config=config_services.manager_config,
            output_paths=output_paths,
            project_root=project_root
        )        
        
        ocr_stager = OCRStager(
            workers=ocr_workers,
            stage_config=config_services.manager_config,
            output_paths=output_paths,
            project_root=project_root
        )
        
        vectorization_stager = VectorizationStager(
            workers=vectorization_workers,
            stage_config=config_services.manager_config,
            output_paths=output_paths,
            project_root=project_root
        )
        
        builder = ProcessingBuilder(
            manager=manager,
            input_stager=input_stager,
            preprocessing_stager=preprocessing_stager,
            ocr_stager=ocr_stager,
            vectorization_stager=vectorization_stager
        )
        builders.append(builder)
        
    return builders

def execute_processing(builders: List['ProcessingBuilder'], workflow_report: Dict[str, Any]) -> Dict[str, Any]:
    """Ejecuta el procesamiento para cada builder."""
    results: Dict[str, Any] = {}
    image_info_list = workflow_report.get('image_info', [])

    for i, builder in enumerate(builders):
        if i < len(image_info_list):
            image_data = image_info_list[i] 
            result = builder.process_single_image()
            results[image_data.get('name', f'imagen_{i}')] = result

    return {
        "mode": workflow_report.get('mode', 'unknown'),
        "processed": len(results),
        "results": results
    }
    