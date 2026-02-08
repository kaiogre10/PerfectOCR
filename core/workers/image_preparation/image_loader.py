# PerfectOCR/core/image_preparation/image_loader.py
import cv2
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any
from core.factory.abstract_worker import ImagePrepAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.utils.image_utils import validate_image, decolorate

logger = logging.getLogger(__name__)

class ImageLoader(ImagePrepAbstractWorker):
    def __init__(self, config: Dict[str, Any], image_data: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.output = config.get("full_img")
        self.image_data = image_data
                        
    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """Carga la imagen y extrae metadatos."""

        image_name = self.image_data.get('name', "")
        input_path = self.image_data.get('full_path', "")
        dpi = self.image_data.get('dpi', {})
        
        metadata: Dict[str, Any] = {
            "image_name": image_name,
            "date_creation": None,
            "dpi": dpi
        }
        try:
            now = datetime.now()
            date_creation = now.isoformat()

            full_image = cv2.imread(input_path, cv2.IMREAD_COLOR).astype(np.uint8)

            if not validate_image(full_image):
                logger.error(f"No se cargó:'{image_name}'")
                return False
                
            full_img = decolorate(full_image)
            
            if self.output:
                from services.output_service import save_croped_image
                output_paths = context["output_paths"]
                worker_name = context.get("worker_name") or "loader"
                img_id = f"full_img_{image_name}_{worker_name}"
                save_croped_image(image_name, img_id, full_img, output_paths, worker_name)

            logger.critical(f"Imagen: '{image_name}' cargada el {now}")
            img_dims = full_img.shape
            
            if not img_dims:
                logger.error(f"Imagen {image_name} totalmente en blanco")
                return False
            
            height, width = img_dims
            size = float(height * width)
            
            logger.debug(f"Dimensiones de la imagen '{image_name}': '{height, width}', size='{size}'")
            
            metadata["date_creation"] = date_creation
                            
            fecha = now.strftime("%Y%m%d")
            decimales = f"{now.microsecond:03d}"
            IDRegistro: str= f"{metadata.get('image_name')}_{fecha}{decimales}"

            if manager.create_workflow(IDRegistro, full_img, metadata):
                logger.debug(f"Imagen '{image_name}' cargada en el manager")
                return True
            else:
                logger.error(f"Error cargando '{image_name}'")
                return False
        except cv2.error as e:
            logger.error(f"Error al cargar la imagen: {image_name}; {e}", exc_info=True)
            return False
