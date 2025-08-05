# PerfectOCR/main.py
import os
import sys
import typer
import logging
from pathlib import Path
from typing import Optional, List
from services.cache_service import CacheService
from services.config_service import ConfigService
from app.process_builder import ProcessingBuilder
from app.workflow_builder import WorkFlowBuilder
from core.pipeline.input_manager import InputManager
from core.pipeline.preprocessing_manager import PreprocessingManager
from core.pipeline.ocr_manager import OCREngineManager

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

def _get_input_paths() -> List[str]:
    """Obtiene rutas de entrada: CLI o por defecto."""
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        # Rutas desde argumentos directos: python main.py /input /input2
        return sys.argv[1:]
    else:
        # Rutas por defecto desde ConfigManager
        temp_config = ConfigService(MASTER_CONFIG_FILE)
        default_input = temp_config._paths_config.get('input_folder')
        return [default_input]

@app.command()
def run(
    input_paths: Optional[List[str]] = typer.Argument(None, help="Rutas de entrada (opcional, usa rutas por defecto)"),
    output_dir: str = typer.Option(None, "--output", "-o", help="Directorio de salida"),
    config: str = typer.Option(MASTER_CONFIG_FILE, "--config", "-c", help="Archivo de configuración"),
    mode: Optional[str] = typer.Option(None, "--mode", "-m", help="Forzar modo: 'interactive' o 'batch'"),
    workers: Optional[int] = typer.Option(None, "--workers", "-w", help="Número de procesos"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Solo mostrar estimación"),
):
    """Ejecuta PerfectOCR con rutas por defecto o personalizadas."""
    
    # Si no se proporcionan rutas, usar las del YAML
    if not input_paths:
        input_paths = _get_input_paths()
    
    # Ejecutar procesamiento
    results = _activate_main(input_paths, output_dir, config, mode, workers, dry_run)
    
    if not dry_run:
        typer.echo(f"Procesamiento completado: {results['processed']} imágenes")
    
    return results

# ========== FUNCIÓN PRINCIPAL DE PROCESAMIENTO ==========
def _activate_main(input_paths: List[str], output_dir: Optional[str], config: str, 
                  mode: Optional[str], workers: Optional[int], dry_run: bool):
    """Función principal de procesamiento - fusiona la lógica original."""
    try:
        # 1. Main activa al Contralor (ConfigManager)
        logging.info("Activando ConfigManager...")
        config_services = ConfigService(config)
        project_root = PROJECT_ROOT

        # 2. Main crea WorkFlowBuilder con configuración centralizada
        logging.info("Creando WorkFlowBuilder...")
        workflow_manager = WorkFlowBuilder(
            config_services=config_services,
            project_root=project_root
        )        
        # 3. WorkflowManager analiza y reporta
        logging.info("Analizando imágenes disponibles...")
        workflow_report = workflow_manager._count_and_plan()
        
        if dry_run:
            return _show_estimation(workflow_report)
        
        # 4. Main crea builders según el reporte
        logging.info("Creando builders según análisis...")
        builders = _create_builders(config_services, project_root, workflow_report)
        
        # 5. Main ejecuta procesamiento
        logging.info("Iniciando procesamiento...")
        results = _execute_processing(builders, workflow_report)
        
        return results
        
    except Exception as e:
        logging.error(f"Error fatal en main: {e}", exc_info=True)
        return {"error": str(e)}
    finally:
        logging.info("Procesamiento finalizado, iniciando limpieza de caché.")
        cache_manager = CacheService(config_services)
        cache_manager.cleanup_project_cache()


def _create_builders(config_services, project_root, workflow_report):
    """Crea builders para cada imagen encontrada."""
    builders = []
    image_info_list = workflow_report.get('image_info', [])
    
    for image_data in image_info_list:
        # Cada builder se asocia con UNA imagen específica
        input_manager = InputManager(
            config=config_services._image_loader_config,
            stage_config=config_services._manager_config,
            input_path=image_data,  # Pasa el dict de la imagen
            project_root=project_root
        )
        
        preprocessing_manager = PreprocessingManager(
            config=config_services._preprocessing_config,
            stage_config=config_services._manager_config,
            project_root=project_root
        )
        
        ocr_manager = OCREngineManager(
            config=config_services._paddle_rec_config,
            stage_config=config_services._manager_config,
            project_root=project_root
        )
        
        builder = ProcessingBuilder(
            input_manager=input_manager,
            preprocessing_manager=preprocessing_manager,
            ocr_manager=ocr_manager
        )
        builders.append(builder)
        
    return builders

def _execute_processing(builders, workflow_report):
    """Ejecuta el procesamiento para cada builder."""
    results = {}
    image_info_list = workflow_report.get('image_info', [])

    for i, builder in enumerate(builders):
        if i < len(image_info_list):
            image_name = image_info_list[i].get('name', f'imagen_{i}')
            result = builder._process_single_image(image_name)
            results[image_name] = result
    
    return {
        "mode": workflow_report.get('mode', 'unknown'),
        "processed": len(results),
        "results": results
    }
    
def _show_estimation(workflow_report: dict):
    """Muestra estimación sin procesar."""
    from utils.batch_tools import estimate_processing_time
    
    total_images = len(workflow_report['image_info'])
    estimation = estimate_processing_time(total_images)
    
    typer.echo(f"Estimación:")
    typer.echo(f"Imágenes encontradas: {total_images}")
    typer.echo(f"Modo: {workflow_report['mode']}")
    typer.echo(f"Tiempo estimado: {estimation['parallel_minutes']:.1f} minutos")
    typer.echo(f"Workers: {estimation['workers']}")
    
    return {"mode": "estimation", "total_images": total_images}


def main():
    """
    Función main original para compatibilidad con ejecución directa.
    """
    # Detectar si hay argumentos CLI
    if len(sys.argv) == 1:
        # Sin argumentos: usar rutas por defecto
        input_paths = _get_input_paths()
        return _activate_main(input_paths, None, MASTER_CONFIG_FILE, None, None, False)
    elif len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        # Argumentos directos: python main.py /input /input2
        input_paths = sys.argv[1:]
        return _activate_main(input_paths, None, MASTER_CONFIG_FILE, None, None, False)
    else:
        app()

if __name__ == "__main__":
    main()