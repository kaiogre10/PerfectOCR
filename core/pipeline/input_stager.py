# PerfectOCR/core/pipeline/input_stager.py
import logging
import time
from typing import Optional, Tuple, List
from core.workers.image_preparation.image_loader import ImageLoader
from core.workers.factory.abstract_worker import AbstractWorker
from core.domain.workflow_job import WorkflowJob

logger = logging.getLogger(__name__)

class InputStager:
    def __init__(self, workers_factory: List[AbstractWorker], image_loader: ImageLoader, project_root: str):
        self.project_root = project_root
        self.workers_factory = workers_factory
        self.image_loader = image_loader

    def generate_polygons(self) -> Tuple[Optional[WorkflowJob], float]:
        start_time = time.time()
        
        workflow_job = self.image_loader.load_image_and_metadata()
        if not workflow_job or workflow_job.full_img is None:
            return None, 0.0
        
        # Contexto con el WorkflowJob para todos los workers
        context = {
            "workflow_job": workflow_job,
            "metadata": {
                "img_dims": {
                    "width": workflow_job.doc_metadata.img_dims.width if workflow_job.doc_metadata else 0,
                    "height": workflow_job.doc_metadata.img_dims.height if workflow_job.doc_metadata else 0
                }
            } if workflow_job.doc_metadata else {}
        }
        current_image = workflow_job.full_img
        
        for worker in self.workers_factory:
            try:
                current_image = worker.process(current_image, context)
                workflow_job.full_img = current_image
            except Exception as e:
                workflow_job.add_error(f"Error en {worker.__class__.__name__}: {e}")
                return None, 0.0
    
        return workflow_job, time.time() - start_time
