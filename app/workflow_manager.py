# PerfectOCR/management/workflow_manager.py
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils.batch_tools import chunked, get_optimal_workers, estimate_processing_time
from management.config_manager import ConfigManager
from app.process_builder import ProcessingBuilder
# Se necesitan estos imports para la reconstrucción en el worker
from core.pipeline.polygon_manager import PolygonManager
from core.pipeline.preprocessing_manager import PreprocessingManager
from core.pipeline.ocr_manager import OCREngineManager

logger = logging.getLogger(__name__)

# --- INICIALIZADOR DE WORKERS PARA MULTIPROCESAMIENTO ---

# Variable global para el worker. Cada proceso tendrá su propia instancia.
processing_builder_instance: Optional[ProcessingBuilder] = None

def _init_worker(config_manager: ConfigManager, project_root: str):
    """
    Función que se ejecuta una vez por cada proceso del pool.
    Reconstruye la cadena de dependencias completa para que el worker sea autónomo.
    """
    global processing_builder_instance
    
    # 1. El worker crea sus propios Jefes de Área (Managers)
    polygon_manager = PolygonManager(config_manager.get_polygonal_config(), project_root)
    preprocessing_manager = PreprocessingManager(config_manager.get_preprocessing_config(), project_root)
    ocr_manager = OCREngineManager(
        config=config_manager.get_ocr_config_with_paths(),
        project_root=project_root,
        output_flags=config_manager.get_enabled_outputs(),
        workflow_config=config_manager.get_processing_config()
    )
    
    # 2. El worker crea su propio Director de Operaciones (ProcessingBuilder)
    local_builder = ProcessingBuilder(
        polygon_manager=polygon_manager,
        preprocessing_manager=preprocessing_manager,
        ocr_manager=ocr_manager,
        output_flags=config_manager.get_enabled_outputs(),
        paths_config=config_manager.get_paths_config()
    )
    
    # 3. Se asigna el builder a la variable global de ESTE worker.
    processing_builder_instance = local_builder
    logger.info(f"Worker (PID: {os.getpid()}) inicializado con su propio ProcessingBuilder.")

def process_image_worker(image_path: str) -> Optional[Dict[str, Any]]:
    """Función que ejecuta cada worker para procesar una imagen."""
    global processing_builder_instance
    if processing_builder_instance is None:
        raise RuntimeError("Worker no inicializado correctamente. El ProcessingBuilder es None.")
    
    try:
        return processing_builder_instance.process_single_image(image_path)
    except Exception as e:
        logger.error(f"Error en worker (PID: {os.getpid()}) procesando {image_path}: {e}", exc_info=True)
        return {"error": str(e), "image": image_path}

# --- GESTOR DE FLUJO DE TRABAJO ---

class WorkflowManager:
    def __init__(self, config_manager: ConfigManager, builder: ProcessingBuilder, project_root: str):
        self.config_manager = config_manager
        self.builder = builder  # Builder para modo interactivo
        self.project_root = project_root # Necesario para el inicializador de workers
        
        self.processing_config = config_manager.get_processing_config()
        self.batch_config = self.processing_config.get('batch_processing', {})
        self.small_batch_limit = self.batch_config.get('small_batch_limit', 5)
        self.max_physical_cores = config_manager.get_max_workers_for_cpu()

    def run(self, force_mode: Optional[str] = None, workers_override: Optional[int] = None) -> Dict[str, Any]:
        """
        Método principal que ejecuta el procesamiento.
        """
        paths_config = self.config_manager.get_paths_config()
        input_folder = paths_config.get('input_folder', './input')
        output_folder = paths_config.get('output_folder', './output')
        valid_extensions = tuple(self.config_manager.get_image_loader_config().get('valid_image_extensions', ['.png']))

        if not os.path.isdir(input_folder):
            logger.critical(f"La carpeta de entrada no existe: {input_folder}")
            return {"error": f"Carpeta de entrada no encontrada: {input_folder}"}

        image_paths = []
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(valid_extensions):
                image_paths.append(os.path.join(input_folder, filename))

        if not image_paths:
            logger.critical("No se encontraron imágenes válidas en la carpeta de entrada.")
            return {"error": "No se encontraron imágenes válidas"}
        
        return self.process_images(image_paths, output_folder, force_mode, workers_override)

    def process_images(self, image_paths: List[str], output_dir: str, force_mode: Optional[str] = None, workers_override: Optional[int] = None) -> Dict[str, Any]:
        num_images = len(image_paths)
        
        if force_mode:
            use_batch = force_mode == 'batch'
        else:
            use_batch = num_images > self.small_batch_limit
            
        estimation = estimate_processing_time(num_images, self.max_physical_cores)
        logger.info(f"Procesando {num_images} imágenes en modo {'LOTE' if use_batch else 'INTERACTIVO'}")
        logger.info(f"Tiempo estimado: {estimation['parallel_minutes']:.1f} min con {estimation['workers']} workers")

        if use_batch:
            return self._process_batch_mode(image_paths, workers_override)
        else:
            return self._process_interactive_mode(image_paths)

    def _process_interactive_mode(self, image_paths: List[str]) -> Dict[str, Any]:
        logger.info("Iniciando modo INTERACTIVO (un solo proceso)")
        results_map = {}
        for i, image_path in enumerate(image_paths, 1):
            logger.info(f"Procesando imagen {i}/{len(image_paths)}: {os.path.basename(image_path)}")
            try:
                result = self.builder.process_single_image(image_path)
                doc_id = os.path.splitext(os.path.basename(image_path))[0]
                results_map[doc_id] = result
            except Exception as e:
                logger.error(f"Error procesando {image_path}: {e}", exc_info=True)
                doc_id = os.path.splitext(os.path.basename(image_path))[0]
                results_map[doc_id] = {"error": str(e), "document_id": doc_id}
        
        return {
            "mode": "interactive",
            "processed": len(results_map),
            "results": results_map
        }

    def _process_batch_mode(self, image_paths: List[str], workers_override: Optional[int] = None) -> Dict[str, Any]:
        logger.info("Iniciando modo LOTE (múltiples procesos)")
        
        workers = workers_override if (workers_override and workers_override > 0) else get_optimal_workers(len(image_paths), self.max_physical_cores)
        logger.info(f"Usando {workers} workers para el procesamiento en lote.")
        
        results_map = {}
        processed_count = 0

        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(self.config_manager, self.project_root) # Pasa los "ingredientes" a cada worker
        ) as executor:
            
            futures = {executor.submit(process_image_worker, path): path for path in image_paths}
            
            for future in as_completed(futures):
                image_path = futures[future]
                doc_id = os.path.splitext(os.path.basename(image_path))[0]
                try:
                    result = future.result()
                    results_map[doc_id] = result
                    processed_count += 1
                    if processed_count % 10 == 0 or processed_count == len(image_paths):
                        logger.info(f"Progreso: {processed_count}/{len(image_paths)} imágenes completadas.")
                except Exception as e:
                    logger.error(f"Error obteniendo resultado para {image_path}: {e}", exc_info=True)
                    results_map[doc_id] = {"error": str(e), "document_id": doc_id}
        
        return {
            "mode": "batch",
            "workers_used": workers,
            "processed": len(results_map),
            "results": results_map
        }
