# core/image_loader.py
import cv2
import numpy as np
import logging
import os
import datetime
from typing import Dict, Any, Tuple, Optional
from PIL import Image
from core.domain.workflow_job import WorkflowJob, DocumentMetadata, ImageDimensions, ProcessingStage
import time

logger = logging.getLogger(__name__)

class ImageLoader:
    """
    Módulo especializado en carga de imágenes y metadatos.
    Responsabilidad única: cargar imagen + extraer metadatos en una sola operación.
    """
    def __init__(self, config: Dict, input_path: Dict, project_root: str):
        self.project_root = project_root
        self.config = config
        self.input_path = input_path

    def _resolutor(self, input_path: Dict) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
        """
        Resuelve la ruta, carga la imagen y extrae metadatos.
        Devuelve (None, metadata_con_error) si falla.
        """
        image_path = input_path.get('path', "")
        image_name = input_path.get('name', "")
        extension = input_path.get('extension', "")
        
        metadata = {
            "image_name": image_name,
            "formato": extension,
            "img_dims": {"width": None, "height": None},
            "dpi": None,
            "error": None
        }

        if not image_path or not os.path.exists(image_path):
            error_msg = f"Ruta de imagen no válida o no encontrada: '{image_path}'"
            logger.error(error_msg)
            metadata['error'] = error_msg
            return None, metadata

        try:
            with Image.open(image_path) as img:
                img.load()
            
            logger.info(f"Imagen '{image_name}' cargada exitosamente desde '{image_path}'")
            
            # Poblar metadatos con información real
            metadata["img_dims"] = {"width": int(img.size[0]), "height": int(img.size[1])}
            dpi_info = img.info.get('dpi')
            if dpi_info:
                metadata["dpi"] = int(dpi_info[0])
            
            return img, metadata
            
        except Exception as e:
            error_msg = f"Error al abrir o leer la imagen '{image_path}': {e}"
            logger.error(error_msg)
            metadata['error'] = error_msg
            return None, metadata

    def _load_image_and_metadata(self, input_path: Dict) -> Tuple[Optional[WorkflowJob], Dict[str, Any]]:
        """
        Carga imagen y extrae metadatos en una sola operación.
        """
        logger.info(f"[ImageLoader] Iniciando carga de imagen: {input_path.get('name', 'desconocido')}")
        start_time = time.time()
        
        img, metadata = self._resolutor(input_path)
        
        if img is None:
            logger.error(f"[ImageLoader] No se pudo resolver la imagen desde: {input_path}")
            return None, metadata

        try:
            image_array = np.array(img)
            
            if len(image_array.shape) == 3:
                gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image_array
            
            # CREAR WORKFLOWJOB AQUÍ
            job_id = f"job_{metadata['image_name']}_{int(time.time())}"
            logger.info(f"[ImageLoader] Creando WorkflowJob con ID: {job_id}")
            
            # Crear ImageDimensions
            img_dims_dict = metadata.get("img_dims", {})
            img_dims = ImageDimensions(
                width=int(img_dims_dict.get("width", 0)),
                height=int(img_dims_dict.get("height", 0))
            )
            
            # Crear DocumentMetadata
            doc_metadata = DocumentMetadata(
                doc_name=metadata["image_name"],
                img_dims=img_dims,
                formato=metadata.get("formato"),
                dpi=metadata.get("dpi"),
                fecha_creacion=datetime.datetime.now()
            )
            
            # Crear WorkflowJob
            workflow_job = WorkflowJob(
                job_id=job_id,
                full_img=gray_image,
                doc_metadata=doc_metadata,
                current_stage=ProcessingStage.IMAGE_LOADED
            )
            
            workflow_job.update_stage(ProcessingStage.IMAGE_LOADED)
            
            load_time = time.time() - start_time
            workflow_job.processing_times["image_loading"] = load_time
            
            logger.info(f"[ImageLoader] WorkflowJob creado exitosamente en {load_time:.3f}s")
            logger.info(f"[ImageLoader] Dimensiones: {img_dims.width}x{img_dims.height}, DPI: {doc_metadata.dpi}")
            
            return workflow_job, metadata

        except Exception as e:
            logger.error(f"[ImageLoader] Error al convertir la imagen a formato numpy: {e}")
            metadata['error'] = f"Error de conversión a numpy: {e}"
            return None, metadata
