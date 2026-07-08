# domain/abstract_worker.py
from abc import ABC, abstractmethod
from typing import Dict, Any
from domain.data_formatter import DataFormatter

class BaseWorker(ABC):
    """Cada Abstract worker es una etapa en el pipeline."""
    __slots__ = ("config", "project_root")
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.config = config
        self.project_root = project_root    

class ImagePrepAbstractWorker(BaseWorker):
    @abstractmethod
    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """Carga y preparada la imagen para su procesamiento"""
        pass
    
class PreprocessingAbstractWorker(BaseWorker):
    @abstractmethod
    def preprocess(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """Recibe la imagen para corregirla si es necesario"""
        pass
    
class OCRAbstractWorker(BaseWorker):
    @abstractmethod
    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """Transcribe la imagen por OCR y limpia/corrije texto"""
        pass
    
class VectorizationAbstractWorker(BaseWorker):
    @abstractmethod
    def vectorize(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """Vectoriza el resultado del texto ocr"""
        pass