# core/image_loader.py
import cv2
import numpy as np
import logging
import os
import datetime
from typing import Dict, Any, Tuple, Optional
from PIL import Image

logger = logging.getLogger(__name__)

class ImageLoader:
    """
    Módulo especializado en carga de imágenes y metadatos.
    Responsabilidad única: cargar imagen + extraer metadatos en una sola operación.
    """
    def __init__(self, config: Dict, input_path: Dict, project_root: str):
        self.project_root = project_root
        self.config = config
        # self.input_path se recibe en __init__ pero es sobrescrito por el que se pasa
        # a _load_image_and_metadata. Es redundante, pero se mantiene la firma por ahora.
        self.input_path = input_path

    def _resolutor(self, input_path: Dict) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
        """
        Resuelve la ruta, carga la imagen y extrae metadatos.
        Devuelve (None, metadata_con_error) si falla.
        """
        # La clave correcta es 'path', no 'full_path'.
        image_path = input_path.get('path', "")
        image_name = input_path.get('name', "")
        extension = input_path.get('extension', "")
        
        # Crear un diccionario de metadatos base para ir llenando
        metadata = {
            "image_name": image_name,
            "formato": extension,
            "img_dims": {"width": None, "height": None},
            "dpi": None,
            "error": None
        }

        if not image_path or not os.path.exists(image_path):
            error_msg = f"Ruta de imagen no válida o no encontrada: '{image_path}'"
            logger.error(error_msg)
            metadata['error'] = error_msg
            return None, metadata

        try:
            with Image.open(image_path) as img:
                img.load()  # Cargar datos de la imagen en memoria
            
            logger.info(f"Imagen '{image_name}' cargada exitosamente desde '{image_path}'")
            
            # Poblar metadatos con información real
            metadata["img_dims"] = {"width": int(img.size[0]), "height": int(img.size[1])}
            dpi_info = img.info.get('dpi')
            if dpi_info:
                metadata["dpi"] = int(dpi_info[0])
            
            return img, metadata
            
        except Exception as e:
            error_msg = f"Error al abrir o leer la imagen '{image_path}': {e}"
            logger.error(error_msg)
            metadata['error'] = error_msg
            return None, metadata

    def _load_image_and_metadata(self, input_path: Dict) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Carga imagen y extrae metadatos en una sola operación.
        """
        img, metadata = self._resolutor(input_path)
        
        if img is None:
            logger.error(f"No se pudo resolver la imagen desde: {input_path}")
            # Devuelve None para la imagen, pero los metadatos (que pueden tener un error)
            return None, metadata

        try:
            # Convertir a numpy array
            image_array = np.array(img)
            
            # Asegurar que está en escala de grises
            if len(image_array.shape) == 3:
                gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image_array
            
            return gray_image, metadata

        except Exception as e:
            logger.error(f"Error al convertir la imagen a formato numpy: {e}")
            metadata['error'] = f"Error de conversión a numpy: {e}"
            return None, metadata
