# PerfectOCR/core/image_preparation/image_loader.py
import cv2
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
from core.factory.abstract_worker import ImagePrepAbstractWorker
from core.domain.data_formatter import DataFormatter

logger = logging.getLogger(__name__)

class ImageLoader(ImagePrepAbstractWorker):
    """
    Módulo especializado en carga de imágenes y metadatos.
    Responsabilidad única: cargar imagen + extraer metadatos en una sola operación.
    """
    def __init__(self, config: Dict[str, Any], image_data: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.image_data = image_data
        self.worker_config = self.config.get('image_loader', {})
        self.enabled_outputs = self.config.get("enabled_outputs", {})
        self.output = self.enabled_outputs.get("pre_clean", False)
        
    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """
        Carga imagen y crea job dict completo usando DataFormatter.
        """
        logger.debug(f"LOADER INCIADO")
        full_img, metadata = self._resolutor(context)
        if full_img is None or ('error' in metadata):
            logger.error(f"Error cargando imagen: {metadata.get('error', 'Unknown error')}")
            
        now = datetime.now()
        fecha = now.strftime("%Y%m%d")
        decimales = f"{now.microsecond:06d}"
        IDRegistro: str= f"{metadata.get('image_name')}_{fecha}{decimales}"
        logger.debug(f"workflow_dict con registro: {IDRegistro}")

        if not manager.create_workflow(IDRegistro, full_img, metadata):
            logger.error("InputStager: Fallo al crear dict_job en el manager.")
            return False

        return True
                
    def _resolutor(self, context: Dict[str, Any]) -> Tuple[Optional[np.ndarray[Any, np.dtype[np.uint8]]], Dict[str, Any]]:
        """Carga la imagen y extrae metadatos.
        Devuelve (None, metadata_con_error) si falla."""
        image_debug = self.image_data
        input_path = image_debug.get('full_path',"")
        image_name = image_debug.get('name', "")
        extension = image_debug.get('extension', "")
        
        if not input_path:
            logger.error(f"Falta la ruta de la imagen ('path') en el contexto para '{image_name}'")
            return None, {}

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
            full_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
            if full_img is None:
                logger.error(f"No se cargó:'{image_name}'")
                return None, metadata
                
            logger.info(f"Imagen: {image_name} cargada correctamente")
            if np.all(full_img == 255):
                logger.error(f"Imagen {image_name} totalmente en blanco")
                return None, {}
            cv2_height, cv2_width = full_img.shape[:2]
            cv2_size: float = full_img.size
            if cv2_size == 0:
                logger.error(f"Imagen vacía o corrupta en '{input_path}'")
                return None, {}
            logger.debug(f"Size de la imagen completa: {cv2_size}")

            metadata["img_dims"] = {
                        "width": float(cv2_width), 
                        "height": float(cv2_height),
                        "size": float(cv2_size)
                    }
            logger.debug(f"Dimensiones imagen:{cv2_width, cv2_height}")
        
            return full_img, metadata

        except Exception as e:
            logger.error(f"Error al  la imagen '{input_path}': {e}", exc_info=True)
            return None, metadata