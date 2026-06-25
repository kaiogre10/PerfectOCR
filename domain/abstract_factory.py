# core/factory/abstract_factory.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Callable, TypeVar, Generic, Optional
from domain.protocol import FactoryComponentProtocol

# T ahora es CUALQUIER clase que cumpla con el protocolo del constructor
T = TypeVar('T', bound=FactoryComponentProtocol)

class AbstractBaseFactory(ABC, Generic[T]):
    """Contrato base para fabricar selectivamente cualquier tipo de componente"""
    def __init__(self, module_config: Dict[str, Any], project_root: str):
        self.module_config = module_config
        self.project_root = project_root
        self.registry = self.create_registry()

    @abstractmethod
    def create_registry(self) -> Dict[str, Callable[[Dict[str, Any]], T]]:
        """Cada fábrica define su propio mapa de inicialización."""
        pass

    def create_components(self, names: List[str], context: Optional[Dict[str, Any]] = None) -> List[T]:
        """Instancia de manera selectiva y en orden los componentes solicitados."""
        components: List[T] = []
        build_context: Dict[str, Any] = context if context is not None else {}

        for name in names:
            if name in self.registry:
                component = self.registry[name](build_context)
                components.append(component)
        return components
