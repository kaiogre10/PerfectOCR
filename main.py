# PerfectOCR/main.py
import os
import sys
import logging
from management.cache_manager import CacheManager
from management.config_manager import ConfigManager
from app.process_builder import ProcessingBuilder
from app.workflow_manager import WorkflowManager
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

def main():
    """
    Punto de entrada: Ensambla la aplicación y delega la ejecución.
    """
    try:
        # 1. Main activa al Contralor (ConfigManager), como siempre.
        logging.info("Activando ConfigManager...")
        config_manager = ConfigManager(MASTER_CONFIG_FILE)
        project_root = config_manager.get_system_config().get('project_root', PROJECT_ROOT)

        # 2. Main ahora CONTRATA y ENTRENA a los Jefes de Área (Managers)
        logging.info("Creando Jefes de Área (Managers)...")
        input_manager = InputManager(
            config=config_manager.get_input_config(),
            paddle_config=config_manager.get_paddle_detection_config(),
            project_root=project_root
        )
        
        preprocessing_manager = PreprocessingManager(
            config=config_manager.get_preprocessing_config(),
            project_root=project_root
        )
        
        ocr_manager = OCREngineManager(
            config=config_manager.get_ocr_config_with_paths(),
            project_root=project_root,
            output_flags=config_manager.get_enabled_outputs(),
            workflow_config=config_manager.get_processing_config()
        )

        # 3. Main contrata al Director de Operaciones (ProcessingBuilder)
        processing_builder = ProcessingBuilder(
            input_manager=input_manager,
            preprocessing_manager=preprocessing_manager,
            ocr_manager=ocr_manager,
            output_flags=config_manager.get_enabled_outputs(),
        )
        
        # 4. Main contrata al Director de Logística (WorkflowManager)
        workflow_manager = WorkflowManager(
            config_manager=config_manager,
            builder=processing_builder,
            project_root=project_root
        )
        
        # 5. Dar orden de inicio al Director de Logística.
        logging.info("Iniciando procesamiento...")
        result = workflow_manager.run() # Ya no se le pasa el builder aquí
        
        return result
        
    except Exception as e:
        logging.error(f"Error fatal en main: {e}", exc_info=True)
        return {"error": str(e)}
    finally:
        # La limpieza de caché se mantiene igual.
        logging.info("Procesamiento finalizado, iniciando limpieza de caché.")
        cache_manager = CacheManager(MASTER_CONFIG_FILE)
        cache_manager.cleanup_project_cache()

if __name__ == "__main__":
    main()
