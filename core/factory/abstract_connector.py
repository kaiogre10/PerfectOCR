from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseConnector(ABC):
    """
    Contrato que todo conector de alto nivel de procesamiento debe cumplir.
    Cada worker es una etapa en el pipeline.
    """
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.config = config
        self.project_root = project_root

class ConnectorAbstractWorker(BaseConnector):
    @abstractmethod
    def transfer(self, context: Dict[str, Any]) -> bool:
        """Envía la información a la plataforma destino"""
        pass
