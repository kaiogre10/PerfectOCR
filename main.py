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
from core.workers.image_preparation.geometry_detector import GeometryDetector

# Configuración de threads (se mantiene como está)
os.environ.update({
    'OMP_NUM_THREADS': '1',        # Conservador para evitar contención
    'MKL_NUM_THREADS': '2',        # Conservador
    'FLAGS_use_mkldnn': '1',       # Mantener (es estable en main thread)
})

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

MASTER_CONFIG_FILE = os.path.join(PROJECT_ROOT, "config", "master_config.yaml")
LOG_FILE_PATH = os.path.join(PROJECT_ROOT, "perfectocr.txt")

def setup_logging():
    """Configura el sistema de logging centralizado."""
    logger_root = logging.getLogger()
    logger_root.setLevel(logging.DEBUG)
    if logger_root.hasHandlers():
        logger_root.handlers.clear()

    formatters = {
        'file': logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(module)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ),
        'console': logging.Formatter('%(levelname)s:%(name)s:%(lineno)d - %(message)s')
    }

    file_handler = logging.FileHandler(LOG_FILE_PATH, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatters['file'])
    file_handler.setLevel(logging.DEBUG)
    logger_root.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatters['console'])
    console_handler.setLevel(logging.INFO)
    logger_root.addHandler(console_handler)
    return logging.getLogger(__name__)

logger = setup_logging()

app = typer.Typer(help="PerfectOCR - Sistema de OCR optimizado")

def get_input_paths() -> List[str]:
    """Obtiene rutas de entrada: CLI o por defecto."""
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        return sys.argv[1:]
    else:
        temp_config = ConfigService(MASTER_CONFIG_FILE)
        default_input = temp_config.paths_config.get('input_folder', "")
        return [default_input]
        
def get_output_paths() -> List[str]:
    """Obtiene rutas de salida: CLI o por defecto."""
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        return sys.argv[1:]
    else:
        tmp_config = ConfigService(MASTER_CONFIG_FILE)
        default_output = tmp_config.paths_config.get('output_folder', "")
        return [default_output]


@app.command()
def run(
    input_paths: Optional[List[str]] = typer.Argument(None, help="Rutas de entrada (opcional, usa rutas por defecto)"),
    output_paths: Optional[List[str]] = typer.Argument(None, help="Rutas de salida (opcional, usa rutas por defecto)"),
    config: str = typer.Option(MASTER_CONFIG_FILE, "--config", "-c", help="Archivo de configuración")
) -> Dict[str, Any]:
    """Ejecuta PerfectOCR con rutas por defecto o personalizadas."""
    
    if not input_paths:
        input_paths = get_input_paths()
    
    if not output_paths:
        output_paths = get_output_paths()

    
    results = activate_main(input_paths, output_paths, config)
        
    return results

def activate_main(input_paths: Optional[List[str]], output_paths: Optional[List[str]], config: str) -> Dict[str, Any]:
    """Función principal de procesamiento - fusiona la lógica original."""
        
        # Si no se proporcionan rutas, usar las del YAML
                
    # Ejecutar procesamiento

    config_services = None
    try:
        # 1. Main activa al Contralor (ConfigManager)
        logging.info("Activando ConfigManager...")
        config_services = ConfigService(config)
        project_root = PROJECT_ROOT

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
    builders = []
    image_info_list = workflow_report.get('image_info', [])
    
    for image_data in image_info_list:
        # Crear workers usando factory (inyección en cascada)
        worker_factory = MainFactory(
            config_services.modules_config,
            project_root=project_root
        )
         
        image_load_factory = worker_factory.get_image_preparation_factory() 
        workers = image_load_factory.create_workers([
            'cleaner', 'angle_corrector', 'line_reconstructor', 'polygon_extractor'],
            context
        )
               
        # Unicas Excepciones
        image_loader = ImageLoader(
            image_info=image_data,
            project_root=project_root,
        )
        
        paddleocr = PaddleOCRWrapper(
            config_dict=config_services.paddle_rec_config,
            project_root=project_root
        )

        paddlepaddle = GeometryDetector(
            paddle_config=config_services.paddle_det_config,
            project_root=project_root
        )

        # Crear managers con workers inyectados
        input_stager = InputStager(
            workers_factory=workers,
            image_loader = image_loader,
            paddlepaddle=paddlepaddle,
            stage_config=config_services.manager_config,
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
    results = {}
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

def main():
    """
    Función main original para compatibilidad con ejecución directa.
    """
    # Detectar si hay argumentos CLI
    if len(sys.argv) == 1:
        # Sin argumentos: usar rutas por defecto
        input_paths = get_input_paths()
        output_paths = get_output_paths()
        return activate_main(input_paths, output_paths, MASTER_CONFIG_FILE)
    elif len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        # Argumentos directos: python main.py /input /input2
        input_paths = sys.argv[1:]
        output_paths = sys.argv[1:]
        return activate_main(input_paths, output_paths, MASTER_CONFIG_FILE)
    else:
        app()

if __name__ == "__main__":
    main()