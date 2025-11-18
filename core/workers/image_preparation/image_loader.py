# PerfectOCR/core/image_preparation/image_loader.py
import cv2
import logging
from datetime import datetime
from typing import Dict, Any
from core.factory.abstract_worker import ImagePrepAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.utils.image_utils import validate_full_image, validate_image

logger = logging.getLogger(__name__)

class ImageLoader(ImagePrepAbstractWorker):
    def __init__(self, config: Dict[str, Any], image_data: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.image_data = image_data
                        
    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """Carga la imagen y extrae metadatos."""

        image_name = self.image_data.get('name', "")
        input_path = self.image_data.get('full_path',"")
        dpi = self.image_data.get('dpi', {})
        
        metadata: Dict[str, Any] = {
            "image_name": image_name,
            "img_dims":{
                    "width": None,
                    "height": None,
                    "size": None
                },
            "date_creation": None,
            "dpi": dpi
        }
        try:
            now = datetime.now()
            date_creation = now.isoformat()

            gray_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

            if not validate_image(gray_img):
                logger.error(f"No se cargó:'{image_name}'")
                return False

            logger.critical(f"Imagen: '{image_name}' cargada el {now}")
            img_dims = validate_full_image(gray_img)
            if not img_dims:
                logger.error(f"Imagen {image_name} totalmente en blanco")
                return False
            
            cv2_height, cv2_width = img_dims
            cv2_size = float(cv2_height * cv2_width)
            
            logger.debug(f"Dimensiones de la imagen '{image_name}': '{cv2_height, cv2_width}', size='{cv2_size}'")

            metadata["img_dims"] = {
                "width": float(cv2_width), 
                "height": float(cv2_height),
                "size": float(cv2_size)
            }
            
            metadata["date_creation"] = date_creation
                            
            fecha = now.strftime("%Y%m%d")
            decimales = f"{now.microsecond:04d}"
            IDRegistro: str= f"{metadata.get('image_name')}_{fecha}{decimales}"

            if manager.create_workflow(IDRegistro, gray_img, metadata): # type: ignore
                logger.debug(f"Imagen '{image_name}' cargada en el manager")
                return True
            else:
                logger.error(f"Error cargando '{image_name}'")
                return False
        except Exception as e:
            logger.error(f"Error al cargar la imagen: {image_name}; {e}", exc_info=True)
            return False
