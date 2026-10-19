# domain/abstract_stager.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from domain.data_formatter import DataFormatter

class AbstractStager(ABC):
    """Clase base abstracta para todos los stagers del pipeline."""
    __slots__ = ("workers", "config")
    def __init__(self, workers: List[Any], config: Dict[str, Any]):
        self.workers = workers
        self.config = config
    
    @abstractmethod
    def execute(self, manager: 'DataFormatter', context: Optional[Dict[str, Any]] = None) -> Optional['DataFormatter']:
        """
        Ejecuta el stage completo.
        Retorna: (manager actualizado o None, tiempo de ejecución)
        """
        pass