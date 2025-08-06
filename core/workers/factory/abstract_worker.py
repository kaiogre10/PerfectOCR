# core/workers/factory/abstract_worker.py
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any

class AbstractWorker(ABC):
    """
    Contrato que todo worker de procesamiento debe cumplir.
    Cada worker es una etapa en el pipeline.
    """
    @abstractmethod
    def process(self, image: np.ndarray[Any, Any], context: Dict[str, Any]) -> np.ndarray[Any, Any]:
        """
        Recibe una imagen, la procesa y devuelve la imagen modificada.
        El 'context' puede llevar datos adicionales si es necesario.
        """
        pass