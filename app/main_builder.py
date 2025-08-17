# PerfectOCR/activate_main.py
import time
t_import0 = time.perf_counter()

from typing import Optional, List, Dict, Any
import logging

t_import1 = time.perf_counter()
from app.process_builder import ProcessingBuilder
print(f"Desde MAIN_BUILDER: Import ProcessingBuilder en {time.perf_counter() - t_import1:.6f}s")

t_import2 = time.perf_counter()
from app.workflow_builder import WorkFlowBuilder
print(f"Desde MAIN_BUILDER: Import WorkFlowBuilder en {time.perf_counter() - t_import2:.6f}s")

t_import3 = time.perf_counter()
from core.pipeline.input_stager import InputStager
print(f"Desde MAIN_BUILDER: Import InputStager en {time.perf_counter() - t_import3:.6f}s")

t_import4 = time.perf_counter()
from core.pipeline.preprocessing_stager import PreprocessingStager
print(f"Desde MAIN_BUILDER: Import PreprocessingStager en {time.perf_counter() - t_import4:.6f}s")

t_import5 = time.perf_counter()
from core.pipeline.ocr_stager import OCRStager
print(f"Desde MAIN_BUILDER: Import OCRStager en {time.perf_counter() - t_import5:.6f}s")

t_import6 = time.perf_counter()
from core.pipeline.vectorization_stager import VectorizationStager
print(f"Desde MAIN_BUILDER: Import VectorizationStager en {time.perf_counter() - t_import6:.6f}s")

t_import7 = time.perf_counter()
from core.factory.main_factory import MainFactory
print(f"Desde MAIN_BUILDER: Import MainFactory en {time.perf_counter() - t_import7:.6f}s")

t_import8 = time.perf_counter()
from core.workers.image_preparation.image_loader import ImageLoader
print(f"Desde MAIN_BUILDER: Import ImageLoader en {time.perf_counter() - t_import8:.6f}s")

t_import9 = time.perf_counter()
from core.domain.data_formatter import DataFormatter
print(f"Desde MAIN_BUILDER: Import DataFormatter en {time.perf_counter() - t_import9:.6f}s")

# t_import10 = time.perf_counter()
# from core.workers.ocr.paddle_wrapper import PaddleOCRWrapper
# print(f"Desde MAIN_BUILDER: Import PaddleOCRWrapper en {time.perf_counter() - t_import10:.6f}s")

t_import11 = time.perf_counter()
from services.config_service import ConfigService
print(f"Desde MAIN_BUILDER: Import ConfigService en {time.perf_counter() - t_import11:.6f}s")

t_import12 = time.perf_counter()
from services.cache_service import cleanup_project_cache
print(f"Desde MAIN_BUILDER: Import cleanup_project_cache en {time.perf_counter() - t_import12:.6f}s")
print(f"Desde MAIN_BUILDER: TIEMPO TOTAL {time.perf_counter() - t_import0:.6f}s")

logger = logging.getLogger(__name__)

def activate_main(input_paths: Optional[List[str]], output_paths: Optional[List[str]], config_path: str, project_root: str) -> Dict[str, Any]:
    
    try:
        t0 = time.perf_counter()
        # 1. Main activa al Condifurador
        logging.debug("Activando ConfigManager con validación robusta...")
        config_services = ConfigService(config_path)
        logging.info(f"ConfigManager iniciado en {time.perf_counter()-t0:.6f}s")
        
        # 2. Main crea WorkFlowBuilder con configuración centralizada
        t1 = time.perf_counter()
        logging.debug("Creando WorkFlowBuilder...")
        workflow_manager = WorkFlowBuilder(
            config_services=config_services,
            project_root=project_root,
            input_paths=input_paths,
        )
        logging.info(f"WorkFlowBuilder completado en {time.perf_counter()-t1:.6f}s")
        
        # 3. WorkflowManager analiza y reporta
        t2 = time.perf_counter()
        logging.debug("Analizando imágenes disponibles...")
        workflow_report = workflow_manager.count_and_plan()
        logging.info(f"Analisis Wrokflow builder completado en {time.perf_counter()-t2:.6f}s")
        
        # 4. Main crea builders según el reporte
        t3 = time.perf_counter()
        logging.debug("Creando builders según análisis")
        builders = create_builders(config_services, project_root, workflow_report, output_paths)
        logging.info(f"Builders creados en {time.perf_counter()-t3:.6f}s")
        
        # 5. Main ejecuta procesamiento
        t4 = time.perf_counter()
        logging.debug("Iniciando procesamiento...")
        results = execute_processing(builders, workflow_report)
        logging.info(f"Procesamiento términado en {time.perf_counter()-t4:.6f}s")
        logging.info(f"Proceso términado en {time.perf_counter()-t0:.6f}s")
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
        ocr_workers = ocr_factory.create_workers(["paddle_wrapper"], context)

        vectorizing_factory = worker_factory.get_vectorizing_factory()

        vectorization_workers = vectorizing_factory.create_workers(
            ["lineal", "dbscan", "table_structurer", "math_max"], 
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