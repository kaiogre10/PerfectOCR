# PerfectOCR/main.py
import os
import sys
import typer
import logging
from typing import Optional, List, Dict, Any
from services.cache_service import CacheService
from services.config_service import ConfigService
from app.process_builder import ProcessingBuilder
from app.workflow_builder import WorkFlowBuilder
from core.pipeline.input_stager import InputStager
from core.pipeline.preprocessing_stager import PreprocessingStager
from core.pipeline.ocr_stager import OCRStager
from core.workers.factory.main_factory import MainFactory
from core.workers.image_preparation.image_loader import ImageLoader
from core.workers.ocr.paddle_wrapper import PaddleOCRWrapper

os.environ.update({
    'OMP_NUM_THREADS': '1',        
    'MKL_NUM_THREADS': '2',
    'FLAGS_use_mkldnn': '1',     
})

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

MASTER_CONFIG_FILE = os.path.join(PROJECT_ROOT, "config", "master_config.yaml")
LOG_FILE_PATH = os.path.join(PROJECT_ROOT, "perfectocr.txt")
DEFAULT_INPUT_PAHT = os.path.join(PROJECT_ROOT, "input")
DEFAULT_OUTPUT_PAHT = os.path.join(PROJECT_ROOT, "output")

app = typer.Typer(help="PerfectOCR - Sistema de OCR optimizado")

def main():
    """
    Función main original para compatibilidad con ejecución directa.
    """
    # Detectar si hay argumentos CLI
    if len(sys.argv) == 1:
        # Sin argumentos: usar rutas por defecto
        input_paths = [DEFAULT_INPUT_PAHT]
        output_paths = [DEFAULT_OUTPUT_PAHT]
        if not input_paths or not output_paths:
            logging.info("Sin rutas default")
            return
        return activate_main(input_paths, output_paths, MASTER_CONFIG_FILE)
    elif len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        input_paths = sys.argv[1:]
        output_paths = sys.argv[1:]
        return activate_main(input_paths, output_paths, MASTER_CONFIG_FILE)
    else:
        app()

@app.command()
def run(
    input_paths: Optional[List[str]] = typer.Argument(None),
    output_paths: Optional[List[str]] = typer.Argument(None),
    config: str = typer.Option(MASTER_CONFIG_FILE, "--config", "-c")
):
    # Determinar el modo de arranque
    modo_config = "default" if config == MASTER_CONFIG_FILE else "custom"
    modo_input = "default" if not input_paths else "cli"
    modo_output = "default" if not output_paths else "cli"

    # Puedes pasar estos flags a activate_main o a cualquier parte del pipeline
    activate_main(
        input_paths=input_paths,
        output_paths=output_paths,
        config_path=config,
    )

def activate_main(input_paths: Optional[List[str]], output_paths: Optional[List[str]], config_path: str) -> Dict[str, Any]:
    """Función principal de procesamiento - CON VALIDACIÓN AUTOMÁTICA."""

    project_root = PROJECT_ROOT
    config_path = MASTER_CONFIG_FILE
    config_services = None
    
    try:
        # 1. Main activa al Contralor (ConfigManager) - AHORA CON VALIDACIÓN
        logging.info("Activando ConfigManager con validación robusta...")
        config_services = ConfigService(config_path)

        # 2. Main crea WorkFlowBuilder con configuración centralizada
        logging.info("Creando WorkFlowBuilder...")
        workflow_manager = WorkFlowBuilder(
            config_services=config_services,
            project_root = PROJECT_ROOT
        )        
        # 3. WorkflowManager analiza y reporta
        logging.info("Analizando imágenes disponibles...")
        workflow_report = workflow_manager.count_and_plan()
        
        # 4. Main crea builders según el reporte
        logging.info("Creando builders según análisis")
        builders = create_builders(config_services, project_root, workflow_report)
        
        # 5. Main ejecuta procesamiento
        logging.info("Iniciando procesamiento...")
        results = execute_processing(builders, workflow_report)
        
        return results
        
    except Exception as e:
        logging.error(f"Error fatal en main: {e}", exc_info=True)
        return {"error": str(e)}
    finally:
        logging.info("Procesamiento finalizado, iniciando limpieza de caché.")
        if config_services is not None:
            cache_manager = CacheService(config_services)
            cache_manager.cleanup_project_cache()

def create_builders(config_services: ConfigService, project_root: str, workflow_report: Dict[str, Any])-> List[ProcessingBuilder]:
    """Crea builders para cada imagen encontrada usando inyecciones en cascada."""
    context = {}
    builders: List[ProcessingBuilder] = []
    image_info_list = workflow_report.get('image_info', [])
    
    for image_data in image_info_list:
        # Crear workers usando factory (inyección en cascada)
        worker_factory = MainFactory(
            config_services.modules_config,
            project_root=project_root
        )
         
        image_load_factory = worker_factory.get_image_preparation_factory() 
        
        context = {
            "paddle_det_config": config_services.validated_paddle_config.models.det_model_dir
        }
        
        workers = image_load_factory.create_workers([
            'cleaner', 'angle_corrector', 'geometry_detector', 'line_reconstructor', 'polygon_extractor'],
            context
        )
               
        image_loader = ImageLoader(
            image_info=image_data,
            project_root=project_root,
        )
        
        paddleocr = PaddleOCRWrapper({
            "config_dict": config_services.validated_paddle_config.models.rec_model_dir},
            project_root=project_root
        )

        # Crear managers con workers inyectados
        input_stager = InputStager(
            workers_factory=workers,
            image_loader = image_loader,
            project_root=project_root
        )
        
        preprocessing_stager = PreprocessingStager(
            config=config_services.preprocessing_config,
            stage_config=config_services.manager_config,
            project_root=project_root
        )
        
        ocr_stager = OCRStager(
            stage_config=config_services.manager_config,
            paddleocr=paddleocr,
            project_root=project_root
        )
        
        builder = ProcessingBuilder(
            input_stager=input_stager,
            preprocessing_stager=preprocessing_stager,
            ocr_stager=ocr_stager
        )
        builders.append(builder)
        
    return builders

def execute_processing(builders: List['ProcessingBuilder'], workflow_report: Dict[str, Any]) -> Dict[str, Any]:
    """Ejecuta el procesamiento para cada builder."""
    results: Dict[str, Any] = {}
    image_info_list = workflow_report.get('image_info', [])

    for i, builder in enumerate(builders):
        if i < len(image_info_list):
            image_data = image_info_list[i]  # dict con name, path, extension
            result = builder.process_single_image(image_data)
            results[image_data.get('name', f'imagen_{i}')] = result

    return {
        "mode": workflow_report.get('mode', 'unknown'),
        "processed": len(results),
        "results": results
    }

if __name__ == "__main__":
    main()