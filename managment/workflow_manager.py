# PerfectOCR/core/utils/workflow_manager.py
import json
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from core.workspace.utils.batch_tools import chunked, get_optimal_workers, estimate_processing_time
from main import PerfectOCRWorkflow
from managment.config_manager import ConfigManager

logger = logging.getLogger(__name__)

# Variable global para workers
workflow_instance = None

def _init_worker(cfg_path: str):
    global workflow_instance
    workflow_instance = PerfectOCRWorkflow(cfg_path)

def process_image_worker(image_path: str) -> Dict[str, Any]:
    """Procesa una imagen usando el workflow del worker."""
    global workflow_instance
    if workflow_instance is None:
        raise RuntimeError("Worker no inicializado correctamente")
    
    try:
        return workflow_instance.run_single_image(image_path)
    except Exception as e:
        logger.error(f"Error en worker procesando {image_path}: {e}")
        return {"error": str(e), "image": image_path}

class WorkflowManager:
    def __init__(self, config_path: str, max_workers_override: int = None):
        
        self.config_path = config_path

        try:
            loader = ConfigManager(config_path)
            batch_config = loader.config.get('batch_processing', {})
            self.small_batch_limit = batch_config.get('small_batch_limit', 5)
            
            if max_workers_override and max_workers_override > 0:
                # El usuario manda
                self.max_physical_cores = max_workers_override
            else:
                self.max_physical_cores = loader.get_max_workers_for_cpu()
                
        except Exception as e:
            logger.warning(f"Error cargando configuración: {e}. Usando valores por defecto.")
            # Valores por defecto conservadores
            self.small_batch_limit = 5
            cpu_count = os.cpu_count() or 4
            self.max_physical_cores = max(1, cpu_count - 2)
        
    def should_use_batch_mode(self, num_images: int) -> bool:
        """Decide si usar modo lote basado en el número de imágenes."""
        return num_images > self.small_batch_limit
    
    def process_images(self, image_paths: List[Path], output_dir: str, force_mode: str = None, workers_override: int = None) -> Dict[str, Any]:
        """Procesa imágenes usando el modo óptimo."""
        num_images = len(image_paths)
        if workers_override and workers_override > 0:
            self.max_physical_cores = workers_override
        if force_mode == 'interactive':
            use_batch = False
        elif force_mode == 'batch':
            use_batch = True
        else:
            use_batch = self.should_use_batch_mode(num_images)
        
        # Mostrar estimación
        estimation = estimate_processing_time(num_images)
        logger.info(f"Procesando {num_images} imágenes en modo {'LOTE' if use_batch else 'INTERACTIVO'}")
        logger.info(f"Tiempo estimado: {estimation['parallel_minutes']:.1f} min con {estimation['workers']} workers")
        
        if use_batch:
            result = self._process_batch_mode(image_paths, output_dir, workers_override)
        else:
            result = self._process_interactive_mode(image_paths, output_dir)
                    
        return result
    
    def _process_interactive_mode(self, image_paths: List[Path], output_dir: str) -> Dict[str, Any]:
        """Modo interactivo: un solo proceso, carga única de modelos."""
        logger.info("Iniciando modo INTERACTIVO")
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        workflow = PerfectOCRWorkflow(self.config_path)
        results_map = {}
        for i, image_path in enumerate(image_paths, 1):
            logger.info(f"Procesando imagen {i}/{len(image_paths)}: {image_path.name}")
            try:
                result = workflow.run_single_image(str(image_path))
                doc_id = result.get("document_id", image_path.stem)
                results_map[doc_id] = result
            except Exception as e:
                logger.error(f"Error procesando {image_path}: {e}", exc_info=True)
                doc_id = image_path.stem
                results_map[doc_id] = {"error": str(e), "document_id": doc_id}
        return {
            "mode": "interactive",
            "processed": len(results_map),
            "results": results_map
        }

    def _process_batch_mode(self, image_paths: List[Path], output_dir: str, workers_override: int = None) -> Dict[str, Any]:
        """Modo lote: múltiples procesos con inicialización optimizada."""
        logger.info("Iniciando modo LOTE")        
        workers = workers_override if (workers_override and workers_override > 0) else get_optimal_workers(len(image_paths), self.max_physical_cores)
        batch_size = workers * 2
        results_map = {}
        processed_count = 0
        
        # Usar funciones globales para evitar problemas de pickling
        with ProcessPoolExecutor(
            max_workers=workers, 
            initializer=_init_worker,
            initargs=(self.config_path,)
        ) as executor:
            
            # Procesar en chunks para controlar memoria
            for chunk in chunked(image_paths, batch_size):
                logger.info(f"Procesando chunk de {len(chunk)} imágenes...")
                
                # Enviar trabajos del chunk actual
                futures = {
                    executor.submit(process_image_worker, str(path)): path 
                    for path in chunk
                }
                
                # Recoger resultados del chunk
                for future in as_completed(futures):
                    image_path = futures[future]
                    doc_id = image_path.stem
                    try:
                        result = future.result()
                        results_map[doc_id] = result
                        processed_count += 1
                        
                        if processed_count % 10 == 0:
                            logger.info(f"Progreso: {processed_count}/{len(image_paths)} imágenes completadas")
                            
                    except Exception as e:
                        logger.error(f"Error obteniendo resultado para {image_path}: {e}", exc_info=True)
                        results_map[doc_id] = {"error": str(e), "document_id": doc_id}
        
        return {
            "mode": "batch",
            "workers_used": workers,
            "batch_size": batch_size,
            "processed": len(results_map),
            "results": results_map
        }