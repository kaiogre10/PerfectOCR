import logging
from typing import Dict, Any
from core.contracts.abstract_worker import ImagePrepAbstractWorker
from domain.data_formatter import DataFormatter
from components.compiled.image.image import load
from utils.text_utils import get_ids

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
            del context["image_data"]
            ptr = load(input_path)

            if not ptr or not isinstance(ptr, int): # type: ignore
                raise TypeError("ERROR DE PUNTEROS")

            image_name = get_ids(input_path, "file_name")

            if not image_name:
                logger.error("No se encontró ninguna coincidencia.")
                return False

            if manager.create_workflow(ptr, image_name):
                logger.info(f"IMAGEN: '{image_name}' cargada exitosamente, Direccion en memoria: '{hex(ptr)}'")
                return True
        
        except Exception as e:
            logger.error(f"Error cargando imagen: {e}", exc_info = True)
        return False