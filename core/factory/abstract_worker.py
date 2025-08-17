# core/workers/workers_factory/abstract_worker.py
from abc import ABC, abstractmethod
from typing import Dict, Any
from core.domain.data_formatter import DataFormatter

class BaseWorker(ABC):
    """Contrato que todo worker de procesamiento debe cumplir.
    Cada worker es una etapa en el pipeline.
    """
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.config = config
        self.project_root = project_root    

class ImagePrepAbstractWorker(BaseWorker):
    @abstractmethod
    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """
        Recibe el dict de datos (no el esquema), lo procesa y devuelve el dict actualizado.
        """
        pass
    
class PreprossesingAbstractWorker(BaseWorker):
    @abstractmethod
    def preprocess(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """
        Recibe la imagen para corregirla si es necesario
        """
        pass
    
class OCRAbstractWorker(BaseWorker):
    @abstractmethod
    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """
        Transcribe la imagen por OCR
        """
        pass
    
class VectorizationAbstractWorker(BaseWorker):
    @abstractmethod
    def vectorize(self, context: Dict[str, Any], manager: DataFormatter) -> object:
        """
        Vectoriza el resultado del texto ocr
        """
        pass