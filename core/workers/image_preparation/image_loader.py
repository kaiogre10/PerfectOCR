# PerfectOCR/core/image_preparation/image_loader.py
import cv2
import time
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
from PIL import Image

logger = logging.getLogger(__name__)

class ImageLoader:
    """
    Módulo especializado en carga de imágenes y metadatos.
    Responsabilidad única: cargar imagen + extraer metadatos en una sola operación.
    """
    def __init__(self, image_info: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.image_info = image_info
                
    def load_image_and_metadata(self) -> Tuple[Optional[np.ndarray[ Any, Any]], Dict[str, Any]]:
        """
        Carga imagen y crea job dict completo usando DataFormatter.
        """
        gray_image, metadata = self._resolutor(self.image_info)
        if gray_image is None or ('error' in metadata):
            logger.error(f"Error cargando imagen: {metadata.get('error', 'Unknown error')}")

        return gray_image, metadata
                
    def _resolutor(self, image_info: Dict[str, Any]) -> Tuple[Optional[np.ndarray[ Any, Any]], Dict[str, Any]]:
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
            "dpi": None,
            "color": None,
        }

        img_array = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if img_array is None:
            error_msg = f"No se pudo leer la imagen en '{input_path}'"
            logger.error(error_msg)
            metadata['error'] = error_msg
            return None, metadata
        logger.debug(f"Imagen cargada correctamente desde: {input_path}")

        image_array = np.array(img_array)
        cv2_size:float = image_array.size
        if cv2_size == 0:
            error_msg = f"Imagen vacía o corrupta en '{input_path}'"
            logger.error(error_msg)
            metadata['error'] = error_msg
            return None, metadata
        
        # logger.info(f"Size de la imagen completa: {cv2_size}")

        cv2_height, cv2_width = image_array.shape[:2]
        if len(image_array.shape) == 3:
            gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image_array
        
        metadata["img_dims"] = {
                    "width": (cv2_width), 
                    "height": (cv2_height),
                    "size": (cv2_size)
                }
        logger.debug(f"Dimensiones imagen:{cv2_width, cv2_height}")
        
        try:
            with Image.open(input_path) as img:

                metadata["color"] = img.mode
                dpi_info: Optional[Dict[str, Optional[float]]] = img.info['dpi']
            if dpi_info and isinstance(dpi_info, tuple) and len(dpi_info) == 2: 
                metadata["dpi"] = float(dpi_info[0]) 
            elif dpi_info and isinstance(dpi_info, (int, float)):
                metadata["dpi"] = float(dpi_info)
            else:
                metadata["dpi"] = None
            logger.info(f"Loader completado en en {time.perf_counter() - start_time:.6f}s para {image_name}")
            logger.debug(f" metadata: {metadata}")
            return gray_image, metadata

        except Exception as e:
            error_msg = f"Error al  la imagen '{input_path}': {e}"
            logger.error(error_msg)
            metadata['error'] = error_msg
            return None, metadata