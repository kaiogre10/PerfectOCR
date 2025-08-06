# core/workers/preprocessing/base_worker.py
from abc import ABC, abstractmethod
import numpy as np

class PreprocessingWorker(ABC):
    """
    Contrato que todo worker de preprocesamiento debe cumplir.
    Cada worker es una etapa en el pipeline.
    """
    @abstractmethod
    def process(self, image: np.ndarray, context: dict) -> np.ndarray:
        """
        Recibe una imagen, la procesa y devuelve la imagen modificada.
        El 'context' puede llevar datos adicionales si es necesario.
        """
        pass