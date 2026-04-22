# core/workers/workers_factory/abstract_stager.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from core.domain.data_formatter import DataFormatter

class AbstractStager(ABC):
    """Clase base abstracta para todos los stagers del pipeline."""
    
    def __init__(
        self, 
        workers: List[Any],
        stage_config: Dict[str, Any], 
        project_root: str
    ):
        self.workers = workers
        self.stage_config = stage_config
        self.project_root = project_root
    
    @abstractmethod
    def execute(self, manager: 'DataFormatter', context: Optional[Dict[str, Any]] = None) -> Tuple[Optional['DataFormatter'], float]:
        """
        Ejecuta el stage completo.
        Retorna: (manager actualizado o None, tiempo de ejecución)
        """
        pass