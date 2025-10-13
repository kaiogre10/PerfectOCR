# PerfectOCR/core/image_preparation/image_loader.py
import cv2
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any
from core.factory.abstract_worker import ImagePrepAbstractWorker
from core.domain.data_formatter import DataFormatter

logger = logging.getLogger(__name__)

class ImageLoader(ImagePrepAbstractWorker):
    """
    Módulo especializado en carga de imágenes y metadatos.
    Responsabilidad única: cargar imagen + extraer metadatos en una sola operación.
    """
    def __init__(self, config: Dict[str, Any], image_data: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.image_data = image_data
        self.worker_config = self.config.get('image_loader', {})
                        
    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """Carga la imagen y extrae metadatos.
        Devuelve (None, metadata_con_error) si falla."""
        logger.debug(f"LOADER INCIADO")
        image_data = self.image_data
        image_name = image_data.get('name', "")
        input_path = image_data.get('full_path',"")
        
        metadata: Dict[str, Any] = {
            "image_name": image_name,
            "img_dims":{
                    "width": None,
                    "height": None,
                    "size": None
                },
            "date_creation": None
        }
        
        full_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if full_img is None:
            logger.error(f"No se cargó:'{image_name}'")
            return False
        
        now = datetime.now()
        date_creation = now.isoformat()
        logger.debug(f"Imagen: {image_name} cargada, {now}")
        if np.all(full_img == 255):
            logger.error(f"Imagen {image_name} totalmente en blanco")
            return False
        
        cv2_height, cv2_width = full_img.shape[:2]
        cv2_size: float = full_img.size
        if cv2_size == 0:
            logger.error(f"Imagen vacía o corrupta en '{input_path}'")
            return False
        
        logger.debug(f"Size de la imagen completa: {cv2_size}")

        metadata["img_dims"] = {
                    "width": float(cv2_width), 
                    "height": float(cv2_height),
                    "size": float(cv2_size)
                }
        
        metadata["date_creation"] = date_creation
        
        logger.debug(f"Dimensiones imagen:{cv2_width, cv2_height}")

        fecha = now.strftime("%Y%m%d")
        decimales = f"{now.microsecond:04d}"
        IDRegistro: str= f"{metadata.get('image_name')}_{fecha}{decimales}"

        if manager.create_workflow(IDRegistro, full_img, metadata):
            return True