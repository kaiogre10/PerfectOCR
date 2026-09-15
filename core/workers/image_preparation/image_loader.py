import logging
from typing import Dict, Any
from core.contracts.abstract_worker import ImagePrepAbstractWorker
from domain.data_formatter import DataFormatter
# from services.output_service import save_croped_image
from components.compiled.image import load

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
            del context["image_data"]
            ptr = load(input_path)

            # image_name, full_image = load_images(input_path)
            # full_image = normalice_image(full_image)
            if not ptr or not isinstance(ptr, int):
                raise TypeError("ERROR DE PUNTEROS")

            image_name = "1"
            if manager.create_workflow(ptr, image_name):
                logger.debug(f"IMAGEN: '{image_name}' cargada exitosamente, Direccion: '{hex(ptr)}'")
                
                # if self.output:
                #     worker_name = context.get("worker_name") or "loader"
                #     save_croped_image(image_name, f"full_img_{image_name}_{worker_name}", full_image)
                return True
        
        except Exception as e:
            logger.error(f"Error cargando imagen: {e}", exc_info = True)
        return False
