# PerfectOCR/core/utils/output_handlers.py
import json
import os
import logging
from typing import Dict, Any, Optional, List
import cv2
import numpy as np
from .encoders import NumpyEncoder

logger = logging.getLogger(__name__)

class OutputHandler:
    """
    Obrero especializado en guardar datos en formato JSON.
    """
    def __init__(self, config: Optional[Dict] = None):
        self.config = config 

    def should_save_output(self, output_type: str) -> bool:
        enabled_outputs = self.config.get("enabled_outputs", {})
        return enabled_outputs.get(output_type, False)
        
    def save(self, data: Dict[str, Any], output_dir: str, file_name_with_extension: str, output_type: str = None) -> Optional[str]:
        """
        Guarda datos en formato JSON en un archivo.
        Crea el directorio de salida si no existe.

        Args:
            data: Los datos a guardar en formato JSON.
            output_dir: El directorio donde se guardará el archivo.
            file_name_with_extension: El nombre del archivo.
            output_type: El tipo de output (e.g., 'ocr_raw', 'reconstructed_lines', etc.)
        """
        if output_type and not self.should_save_output(output_type):
            logger.debug(f"Output {output_type} está deshabilitado, omitiendo guardado.")
            return None
            
        # Validar que output_dir no esté vacío
        if not output_dir or output_dir.strip() == "":
            logger.error("Error: output_dir está vacío o es None")
            return None

        output_path = None
        try:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, file_name_with_extension)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
            return output_path
        except Exception as e:
            error_msg = f"Error guardando JSON"
            if output_path:
                error_msg += f" en {output_path}"
            error_msg += f": {e}"
            logger.error(error_msg, exc_info=True)
            return None
            
class ImageOutputHandler:
    """Obrero especializado en guardar imágenes."""
    def __init__(self, config: Optional[Dict] = None):
        self.config = config if config is not None else {}

    def save(self, image_array: np.ndarray, output_dir: str, file_name_with_extension: str) -> Optional[str]:
        """
        Guarda un array de numpy como un archivo de imagen.
        """
        output_path = ""
        try:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, file_name_with_extension)
            cv2.imwrite(output_path, image_array)
            return output_path
        except Exception as e:
            logger.error(f"Error guardando imagen en {output_path}: {e}", exc_info=True)
            return None
            
class TextOutputHandler:
    """
    Obrero especializado en guardar contenido de texto plano.
    """
    def __init__(self, config: Optional[Dict] = None):
        self.config = config if config is not None else {}
        # logger.debug("TextOutputHandler initialized.")

    def save(self, text_content: str, output_dir: str, file_name_with_extension: str) -> Optional[str]:
        """
        Guarda una cadena de texto en un archivo.
        Crea el directorio de salida si no existe.

        Args:
            text_content: La cadena de texto a guardar.
            output_dir: El directorio donde se guardará el archivo.
            file_name_with_extension: El nombre del archivo (e.g., "transcription.txt").

        Returns:
            La ruta completa al archivo guardado si tiene éxito, None en caso contrario.
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, file_name_with_extension)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            logger.info(f"Datos de texto guardados en: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error guardando archivo de texto en {output_path}: {e}", exc_info=True)
            return None
        