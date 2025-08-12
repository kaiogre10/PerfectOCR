# core/workers/factory/abstract_factory.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Callable, TypeVar, Generic
from core.factory.abstract_worker import BaseWorker

T = TypeVar('T', bound=BaseWorker)

class AbstractBaseFactory(ABC, Generic[T]):
    """Factory base para todos los módulos."""
    
    def __init__(self, module_config: Dict[str, Any], project_root: str):
        self.module_config = module_config
        self.project_root = project_root
        self.worker_registry = self.create_worker_registry()
        
    
    @abstractmethod
    def create_worker_registry(self) -> Dict[str, Callable[[Dict[str, Any]], T]]:
        """Cada módulo define su registro de workers."""
        pass
    
    def create_workers(self, worker_names: List[str], context: Dict[str, Any]) -> List[T]:
        """Crea workers en el orden especificado."""
        workers: List[T] = []
        for worker_name in worker_names:
            if worker_name in self.worker_registry:
                worker = self.worker_registry[worker_name](context)
                workers.append(worker)
        return workers
    
        
        