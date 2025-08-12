# core/workers/workers_factory/abstract_worker.py
from abc import ABC, abstractmethod
from typing import Dict, Any
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import CroppedImage

class BaseWorker(ABC):
    """Clase base común para los workers abstractos"""
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.config = config
        self.project_root = project_root    

class ImagePrepAbstractWorker(BaseWorker):
    """
    Contrato que todo worker de procesamiento debe cumplir.
    Cada worker es una etapa en el pipeline.
    """
    
    @abstractmethod
    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """
        Recibe el dict de datos (no el esquema), lo procesa y devuelve el dict actualizado.
        """
        pass
    
    
class PreprossesingAbstractWorker(BaseWorker):
    @abstractmethod
    def preprocess(self, cropped_img: CroppedImage, manager: DataFormatter) -> CroppedImage:
        """
        Recibe la imagen para corregirla si es necesario
        """
        pass