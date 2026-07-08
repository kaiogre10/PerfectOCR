import cv2
import logging
import numpy as np
import time
import os
from datetime import datetime
from typing import Dict, Any
from domain.abstract_worker import ImagePrepAbstractWorker
from domain.data_formatter import DataFormatter
from utils.image_utils import decolorate, is_binarized
from services.output_service import save_croped_image

logger = logging.getLogger(__name__)

class ImageLoader(ImagePrepAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.output = config.get("full_img")
                        
    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """Carga la imagen y extrae metadatos."""
        try:
            input_path = context.get("image_data", "")
            image_name = os.path.splitext(os.path.basename(input_path))[0]

            if not input_path or not os.path.isfile(input_path):
                raise FileNotFoundError(f"No se proporcionó una ruta de entrada válida: '{input_path}'")
            
            # time0 = time.perf_counter()
            full_image = cv2.imread(input_path, cv2.IMREAD_COLOR)
            logger.info(f"IMAGEN: '{input_path}'")
                        # "\n"f"cargada en {time.perf_counter() - time0:.6f}'s")
            
            if is_binarized(full_image):
                binary = True
                full_img = full_image
            else:
                full_img = decolorate(full_image)
                binary = False
            
            if full_img.size < 1 or full_img is None: # type: ignore
                return False
            
            full_img = np.require(full_img, dtype=np.uint8, requirements=['C', 'A', 'W', 'O', 'E'])

            # Metadata: una sola llamada a datetime
            now = datetime.now()
            date_creation = f"{now.strftime('%Y%m%d')}"
            metadata: Dict[str, Any] = {
                "image_name": image_name,
                "date_creation": date_creation,
                "binary": binary
            }
            
            id_registro = f"{image_name}_{date_creation}{now.microsecond:08d}"

            if manager.create_workflow(id_registro, full_img, metadata):
                logger.debug(f"IMAGEN: '{image_name}' cargada en workflow exitosamente")
                
                if self.output:
                    worker_name = context.get("worker_name") or "loader"
                    save_croped_image(image_name, f"full_img_{image_name}_{worker_name}", full_img)

                del context["image_data"]
                return True
            
        except Exception as e:
            logger.error(f"Error cargando: {e}", exc_info=True)
        return False
