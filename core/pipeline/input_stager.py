# PerfectOCR/core/pipeline/input_stager.py
import logging
import time
from typing import Any, Optional, Dict, Tuple, List
from core.workers.image_preparation.image_loader import ImageLoader
from core.workers.factory.abstract_worker import AbstractWorker
from core.domain.workflow_job import WorkflowJob, ProcessingStage

logger = logging.getLogger(__name__)

class InputStager:
    """
    Coordina la fase de extracción de polígonos usando inyecciones en cascada.
    Recibe workers ya configurados y solo coordina el flujo.
    """
    def __init__(self, workers_factory: List[AbstractWorker], image_loader: ImageLoader, stage_config: Dict[str, Any], project_root: str):
        self.workers_factory = workers_factory
        self.image_loader = image_loader
        self.stage_config = stage_config
        self.project_root = project_root
        logger.info(f"[InputStager] Inicializado con {len(workers)} workers")

    def generate_polygons(self) -> Tuple[Optional[WorkflowJob], float]:
        """
        Ejecuta el pipeline de workers en orden para generar polígonos.
        """
        pipeline_start = time.time()
        
        try:
            # Lazy loading: ImageLoader crea WorkflowJob completo
            workflow_job = self.image_loader.load_image_and_metadata
            
            if workflow_job is None or workflow_job.full_img is None:
                logger.error("[InputStager] No se pudo cargar la imagen")
                return None, 0.0
            
            # Contexto con el WorkflowJob para todos los workers
            context = {
                "workflow_job": workflow_job,
                "metadata": workflow_job.doc_metadata.__dict__ if workflow_job.doc_metadata else {}
            }
            
            # Ejecutar workers en orden
            current_image = workflow_job.full_img
            
            for worker in self.workers:
                try:
                    logger.info(f"[InputStager] Ejecutando worker: {worker.__class__.__name__}")
                    worker_start = time.time()
                    
                    # Cada worker implementa process() y maneja su lógica específica
                    current_image = worker.process(current_image, context)
                    
                    # Actualizar la imagen en el WorkflowJob
                    workflow_job.full_img = current_image
                    
                    worker_time = time.time() - worker_start
                    workflow_job.processing_times[f"{worker.__class__.__name__}"] = worker_time
                    
                    logger.info(f"[InputStager] Worker {worker.__class__.__name__} completado en {worker_time:.3f}s")
                    
                except Exception as e:
                    error_msg = f"Error en worker {worker.__class__.__name__}: {e}"
                    logger.error(error_msg, exc_info=True)
                    workflow_job.add_error(error_msg)
                    return None, 0.0
            
            # Pipeline completado exitosamente
            total_time = time.time() - pipeline_start
            workflow_job.processing_times["input_pipeline_total"] = total_time
            workflow_job.update_stage(ProcessingStage.POLYGONS_EXTRACTED)
            
            logger.info(f"[InputStager] Pipeline completado exitosamente en {total_time:.3f}s")
            return workflow_job, total_time
            
        except Exception as e:
            error_msg = f"Error fatal en InputStager: {e}"
            logger.error(error_msg, exc_info=True)
            return None, 0.0
