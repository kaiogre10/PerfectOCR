import logging
from typing import Dict, Any
from core.contracts.abstract_worker import ImagePrepAbstractWorker
from domain.data_formatter import DataFormatter
from utils.image_utils import normalice_image
from utils.file_handler import load_images
from services.output_service import save_croped_image
from components.compiled.image import load, get_array, release # type: ignore

logger = logging.getLogger(__name__)

class ImageLoader(ImagePrepAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        # self.project_root = project_root
        self.output = config.get("full_img")

    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """Carga la imagen y extrae metadatos."""
        try:
            input_path = context.get("image_data", "")
            logger.info(f"IMAGEN: '{input_path}'")
            ptr = load(input_path)
            del context["image_data"]
            
            
            image_name, full_image = load_images(input_path)
            full_image = normalice_image(full_image)
            if full_image is None:
                raise TypeError("NO SE PUDO NORMALIZAR LA IMAGEN")
            
            if manager.create_workflow(full_image, image_name):
                logger.debug(f"IMAGEN: '{image_name}' cargada en workflow exitosamente")
                
                if self.output:
                    imagememmory = get_array(ptr)
                    release(ptr)
                    worker_name = context.get("worker_name") or "loader"
                    save_croped_image(image_name, f"full_img_{image_name}_{worker_name}", imagememmory)
                return True
        
        except Exception as e:
            logger.error(f"Error cargando imagen: {e}", exc_info = True)
        return False
