# core/image_loader.py
import cv2
import numpy as np
import logging
import datetime
from typing import Dict, Any, Tuple, Optional
from PIL import Image
from core.domain.workflow_job import WorkflowJob, DocumentMetadata, ImageDimensions
import time

logger = logging.getLogger(__name__)

class ImageLoader:
    """
    Módulo especializado en carga de imágenes y metadatos.
    Responsabilidad única: cargar imagen + extraer metadatos en una sola operación.
    """
    def __init__(self, image_info: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.image_info = image_info

    def resolutor(self, image_info: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Dict[str, Any]]: # type: ignore
        """
        Carga la imagen y extrae metadatos.
        Devuelve (None, metadata_con_error) si falla.
        """
        input_path = image_info['path']
        image_name = image_info.get('name', "")
        extension = image_info.get('extension', "")

        metadata = {
            "image_name": image_name,
            "formato": extension,
            "img_dims": {"width": None, "height": None},
            "dpi": None,
            "mode": None,
        }

        try:
            img_array = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
            image_array = np.array(img_array)
            cv2_height, cv2_width = image_array.shape[:2]
            if len(image_array.shape) == 3:
                gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image_array

            # Extraer metadatos completos con Pillow
            with Image.open(input_path) as img:
                pillow_width, pillow_height = img.width, img.height
                metadata["img_dims"] = {"width": pillow_width, "height": pillow_height}
                metadata["mode"] = img.mode
                dpi_info = img.info.get('dpi')
                if dpi_info and isinstance(dpi_info, tuple) and len(dpi_info) == 2:
                    metadata["dpi"] = float(dpi_info[0])  # Usar DPI X (horizontal)
                elif dpi_info and isinstance(dpi_info, (int, float)):
                    metadata["dpi"] = float(dpi_info)
                else:
                    metadata["dpi"] = None

            # Comprobación de dimensiones
            if (pillow_width == cv2_width) and (pillow_height == cv2_height):
                metadata["img_dims"] = {"width": pillow_width, "height": pillow_height}
            else:
                metadata["img_dims"] = {"width": cv2_width, "height": cv2_height}
                logger.warning(f"Dimensiones distintas: Pillow ({pillow_width}, {pillow_height}) != cv2 ({cv2_width}, {cv2_height}), usando cv2.")

            logger.info(f"Imagen '{image_name}' cargada exitosamente desde '{input_path}'")
            return gray_image, metadata

        except Exception as e:
            error_msg = f"Error al procesar la imagen '{input_path}': {e}"
            logger.error(error_msg)
            metadata['error'] = error_msg
            return None, metadata
    
    def load_image_and_metadata(self) -> WorkflowJob:
        """
        Carga imagen y extrae metadatos en una sola operación.
        """
        image_info = self.image_info
        
        start_time = time.time()
        
        gray_image, metadata = self.resolutor(image_info) # type: ignore
                    
        # CREAR WORKFLOWJOB AQUÍ
        job_id = f"job_{metadata['image_name']}_{int(time.time())}"
                
        # Crear ImageDimensions
        img_dims_dict = metadata.get("img_dims", {})
        img_dims = ImageDimensions(
            width=int(img_dims_dict.get("width")),
            height=int(img_dims_dict.get("height"))
        )
        
        # Crear DocumentMetadata
        doc_metadata = DocumentMetadata(
            doc_name=metadata["image_name"],
            img_dims=img_dims,
            formato=metadata.get("formato"),
            dpi=metadata.get("dpi"),
            color=metadata.get("mode"),
            date_creation=datetime.datetime.now()
        )
        
        # Crear WorkflowJob con TODO incluido
        workflow_job = WorkflowJob(
            job_id=job_id,
            full_img=gray_image,
            doc_metadata=doc_metadata, 
        )
        
        load_time = time.time() - start_time
        workflow_job.processing_times["image_loading"] = load_time
        
        logger.info(f"[ImageLoader] WorkflowJob creado exitosamente en {load_time:.3f}s")
                
        return workflow_job  