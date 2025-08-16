# PerfectOCR/activate_main.py
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
from core.workers.ocr.paddle_wrapper import PaddleOCRWrapper
from services.config_service import ConfigService
from services.cache_service import cleanup_project_cache

logger = logging.getLogger(__name__)

def activate_main(input_paths: Optional[List[str]], output_paths: Optional[List[str]], config_path: str, project_root: str) -> Dict[str, Any]:
    
    try:        
        # 1. Main activa al Condifurador
        logging.info("Activando ConfigManager con validación robusta...")
        config_services = ConfigService(config_path)
    
        # 2. Main crea WorkFlowBuilder con configuración centralizada
        logging.info("Creando WorkFlowBuilder...")
        workflow_manager = WorkFlowBuilder(
            config_services=config_services,
            project_root=project_root,
            input_paths=input_paths,
        )        
        # 3. WorkflowManager analiza y reporta
        logging.info("Analizando imágenes disponibles...")
        workflow_report = workflow_manager.count_and_plan()
        
        # 4. Main crea builders según el reporte
        logging.info("Creando builders según análisis")
        builders = create_builders(config_services, project_root, workflow_report, output_paths)
        
        # 5. Main ejecuta procesamiento
        logging.info("Iniciando procesamiento...")
        results = execute_processing(builders, workflow_report)
        return results
        
    except Exception as e:
        logging.error(f"Error fatal en main: {e}", exc_info=True)
        return {"error": str(e)}

    finally:        
        try:
            logging.info("Procesamiento finalizado, iniciando limpieza de caché.")
            cleanup_project_cache(project_root)
            logging.info("Limpieza de caché completada.")
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
    
        geometry_detector = config_services.paddle_det_config

        context = {
            "geometry_detector": geometry_detector,
        }
        # Obtener factories del worker_factory
        image_load_factory = worker_factory.get_image_preparation_factory()
        image_prep_workers = image_load_factory.create_workers([
            "cleaner", "angle_corrector", "geometry_detector", "polygon_extractor"
        ], context)
        
        # Crear workers SEPARADOS
        preprocessing_factory = worker_factory.get_preprocessing_factory()
        preprocessing_workers = preprocessing_factory.create_workers(
            ["moire", "sp", "gauss", "clahe", "sharp", "binarization", "fragmentator"],
            context
        )
        
        vectorizing_factory = worker_factory.get_vectorizing_factory()
        vectorization_workers = vectorizing_factory.create_workers(
            ["lineal", "dbscan"], 
            context
        )
                
        manager = DataFormatter()
        
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
        
        paddleocr = PaddleOCRWrapper(
            {"paddle_config": config_services.paddleocr},
            project_root=project_root
        )
        
        ocr_stager = OCRStager(
            stage_config=config_services.manager_config,
            paddleocr=paddleocr,
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