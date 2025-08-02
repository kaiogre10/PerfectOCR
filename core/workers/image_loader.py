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
    
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.config = config
        self.project_root = project_root
    
    def load_image_and_metadata(self, input_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Carga imagen y extrae metadatos en una sola operación.
        Elimina la triple transferencia del pipeline actual.
        """
        try:
            # Validar archivo
            self._validate_file(input_path)
            
            # Cargar imagen y metadatos en una sola operación
            with Image.open(input_path) as img:
                # Extraer metadatos
                metadata = self._extract_metadata(img, input_path)
                
                # Convertir a numpy array
                image_array = np.array(img)
                if len(image_array.shape) == 3:
                    # PIL usa RGB, OpenCV usa BGR
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                
                return image_array, metadata
                
        except Exception as e:
            logger.error(f"Error cargando {input_path}: {e}")
            raise ValueError(f"No se pudo cargar {input_path}: {e}")
    
    def _validate_file(self, input_path: str) -> None:
        """Valida que el archivo existe y es una imagen válida."""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Archivo no encontrado: {input_path}")
        
        if not os.path.isfile(input_path):
            raise ValueError(f"No es un archivo válido: {input_path}")
        
        # Validar extensión
        valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
        if not input_path.lower().endswith(valid_extensions):
            raise ValueError(f"Formato de imagen no soportado: {input_path}")
    
    def _extract_metadata(self, img: Image.Image, input_path: str) -> Dict[str, Any]:
        """Extrae metadatos de imagen ya cargada."""
        try:
            # Dimensiones
            img_dims = {
                "width": int(img.size[0]),
                "height": int(img.size[1])
            }
            
            # DPI
            dpi = img.info.get('dpi')
            dpi_val = int(dpi[0]) if dpi is not None else None
            
            # Fecha de creación
            fecha_creacion = None
            try:
                stat = os.stat(input_path)
                fecha_creacion = datetime.datetime.fromtimestamp(
                    stat.st_ctime
                ).strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                pass
            
            return {
                "doc_name": os.path.basename(input_path),
                "formato": img.format,
                "img_dims": img_dims,
                "dpi": dpi_val,
                "fecha_creacion": fecha_creacion
            }
            
        except Exception as e:
            logger.warning(f"Error extrayendo metadatos de {input_path}: {e}")
            return {
                "doc_name": os.path.basename(input_path),
                "formato": None,
                "img_dims": {"width": 0, "height": 0},
                "dpi": None,
                "fecha_creacion": None
            }