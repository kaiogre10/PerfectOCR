import logging
from typing import Dict, Any
from core.contracts.abstract_worker import ImagePrepAbstractWorker
from domain.data_formatter import DataFormatter
from domain.class_models import TypeModels
from bidings.image import load

logger = logging.getLogger(__name__)

class ImageLoader(ImagePrepAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.output = config.get("full_img")

    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """Carga la imagen y extrae metadatos."""
        try:
            input_path = context.get("image_data", "")
            if input_path.isascii():
                input_path = input_path.encode(TypeModels.ASCII.value)
            else:
                raise ValueError(f"NOMBRE DE ARCHIVO CON CARACTERES NO VALIDOS: {input_path}")
            
            del context["image_data"]
            ptr = load(input_path)

            if not ptr or not isinstance(ptr, int): # type: ignore
                raise TypeError("ERROR DE PUNTEROS")

            image_name = context.get("image_name", "")
            if not image_name:
                logger.error("No se encontró ninguna coincidencia.")
                return False

            if manager.create_workflow(ptr, image_name):
                logger.debug(f"IMAGEN: '{image_name}' cargada exitosamente, Direccion en memoria: '{ptr}'")
                return True
        
        except Exception as e:
            logger.exception(f"Error cargando imagen: {e}", exc_info = True)
        return False