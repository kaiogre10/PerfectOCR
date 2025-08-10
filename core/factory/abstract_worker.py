# core/workers/workers_factory/abstract_worker.py
from abc import ABC, abstractmethod
from typing import Dict, Any
from core.domain.data_manager import DataFormatter

class AbstractWorker(ABC):
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