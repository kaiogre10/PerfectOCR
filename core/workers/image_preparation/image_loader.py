import cv2
import logging
import time
from datetime import datetime
from typing import Dict, Any, Set
from core.factory.abstract_worker import ImagePrepAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.utils.image_utils import decolorate
from services.output_service import save_croped_image

logger = logging.getLogger(__name__)

class ImageLoader(ImagePrepAbstractWorker):
    def __init__(self, config: Dict[str, Any], image_data: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.valid_extensions: Set[str] = set(config["valid_image_extensions"])
        self.output = config.get("full_img")
        self.image_data = image_data
                        
    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """Carga la imagen y extrae metadatos."""
        image_info = context.get("image_data", self.image_data)
        
        # Obtener los datos con claves seguras en caso de archivos pasados explícitamente vs por carpeta
        image_name = image_info.get('name', "")
        input_path = image_info.get('full_path', image_info.get('path', ""))
        extension = image_info.get('extension', "").lower()
        
        try:
            if not input_path:
                logger.error(f"No se proporcionó una ruta de entrada válida para la imagen '{image_name}'")
                return False
            # Carga condicional según formato (extension ya viene del builder)
            if extension in self.valid_extensions:
                # cv2.imread ya retorna uint8, no necesita .astype()
                time0 = time.perf_counter()
                full_image = cv2.imread(input_path, cv2.IMREAD_COLOR)
                logger.debug(f"IMAGEN: '{image_name}' cargada en {time.perf_counter() - time0:.4f}'s")
            else:
                logger.error(f"Formato de imagen no válida: {image_name}")
                return False

            if full_image is None:
                logger.error(f"No se pudo cargar: '{image_name}{extension}'")
                return False
                
            full_img = decolorate(full_image)
            
            # Metadata: una sola llamada a datetime
            now = datetime.now()
            metadata: Dict[str, Any] = {
                "image_name": image_name,
                "extension": extension,
                "date_creation": now.isoformat()
            }
            
            IDRegistro = f"{image_name}_{now.strftime('%Y%m%d')}{now.microsecond:04d}"

            if manager.create_workflow(IDRegistro, full_img, metadata):
                logger.info(f"IMAGEN: '{image_name}' cargada en workflow exitosamente")
            
                if self.output:
                    output_paths = context["output_paths"]
                    worker_name = context.get("worker_name") or "loader"
                    save_croped_image(image_name, f"full_img_{image_name}_{worker_name}", full_img, output_paths, worker_name)

                return True
            
            logger.error(f"Error creando workflow para '{image_name}'")
            return False

        except cv2.error as e:
            logger.error(f"Error OpenCV al cargar '{image_name}': {e}", exc_info=True)
        return False
