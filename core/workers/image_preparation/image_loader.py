import cv2
import logging
import time
import numpy as np
from datetime import datetime
from typing import Dict, Any
from core.factory.abstract_worker import ImagePrepAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.utils.image_utils import decolorate, is_binarized
from core.utils.text_utils import get_ids
from services.output_service import save_croped_image

logger = logging.getLogger(__name__)

class ImageLoader(ImagePrepAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.output = config.get("full_img")
                        
    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """Carga la imagen y extrae metadatos."""
        image_info = context.get("image_data", {})
        
        # Obtener los datos con claves seguras en caso de archivos pasados explícitamente vs por carpeta
        image_name = image_info.get('name', "")
        image_name = get_ids(image_name, "name")

        input_path = image_info.get('full_path', image_info.get('path', ""))
        
        try:
            if not input_path:
                logger.error(f"No se proporcionó una ruta de entrada válida para la imagen '{image_name}'")
                return False

            time0 = time.perf_counter()
            full_image = cv2.imread(input_path, cv2.IMREAD_COLOR)
            logger.info(f"IMAGEN: '{input_path}', cargada en {time.perf_counter() - time0:.6f}'s")
            
            if is_binarized(full_image):
                binary = True
                full_img = full_image
            else:
                full_img = decolorate(full_image)
                binary = False
            
            if full_img.size < 1 or full_img is None:
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
            
            id_registro = f"{image_name}_{date_creation}{now.microsecond:06d}"

            if manager.create_workflow(id_registro, full_img, metadata):
                # logger.info(f"IMAGEN: '{image_name}' cargada en workflow exitosamente")
            
                if self.output:
                    worker_name = context.get("worker_name") or "loader"
                    save_croped_image(image_name, f"full_img_{image_name}_{worker_name}", full_img, worker_name)

                return True
            
            logger.error(f"Error creando workflow para '{image_name}'")
            return False

        except cv2.error as e:
            logger.error(f"Error OpenCV al cargar '{image_name}': {e}", exc_info=True)
        return False
