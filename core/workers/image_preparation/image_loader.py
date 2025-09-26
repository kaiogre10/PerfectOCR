# PerfectOCR/core/image_preparation/image_loader.py
import cv2
import time
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class ImageLoader:
    """
    Módulo especializado en carga de imágenes y metadatos.
    Responsabilidad única: cargar imagen + extraer metadatos en una sola operación.
    """
    def __init__(self, image_info: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.image_info = image_info
                
    def load_image_and_metadata(self) -> Tuple[Optional[np.ndarray[Any, np.dtype[np.uint8]]], Dict[str, Any]]:
        """
        Carga imagen y crea job dict completo usando DataFormatter.
        """
        gray_image, metadata = self._resolutor(self.image_info)
        if gray_image is None or ('error' in metadata):
            logger.error(f"Error cargando imagen: {metadata.get('error', 'Unknown error')}")

        return gray_image, metadata
                
    def _resolutor(self, image_info: Dict[str, Any]) -> Tuple[Optional[np.ndarray[Any, np.dtype[np.uint8]]], Dict[str, Any]]:
        """Carga la imagen y extrae metadatos.
        Devuelve (None, metadata_con_error) si falla."""
        start_time = time.perf_counter()
        input_path = image_info['path']
        image_name = image_info.get('name', "")
        extension = image_info.get('extension', "")

        metadata: Dict[str, Any] = {
            "image_name": image_name,
            "format": extension,
            "img_dims":{
                    "width": None,
                    "height": None,
                    "size": None
                },
        }
        try:
            image_array = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if image_array is None:
                logger.error(f"No se cargó:'{image_name}', {e}", exc_info=True)
                return None, metadata
            logger.debug(f"Imagen cargada correctamente desde: {input_path}")
            cv2_height, cv2_width = image_array.shape[:2]
            cv2_size: float = image_array.size
            if cv2_size == 0:
                logger.error(f"Imagen vacía o corrupta en '{input_path}'")
                return None, {}
            logger.debug(f"Size de la imagen completa: {cv2_size}")

            metadata["img_dims"] = {
                        "width": (cv2_width), 
                        "height": (cv2_height),
                        "size": (cv2_size)
                    }
            logger.debug(f"Dimensiones imagen:{cv2_width, cv2_height}")
            logger.debug(f"Loader completado en en {time.perf_counter() - start_time:.6f}s para {image_name}")
        
            return image_array, metadata

        except Exception as e:
            logger.info(f"Error al  la imagen '{input_path}': {e}", exc_info=True)
        return None, metadata