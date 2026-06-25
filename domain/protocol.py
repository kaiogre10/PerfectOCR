# core/factory/protocol.py
from typing import Protocol, Dict, Any

class FactoryComponentProtocol(Protocol):
    """Cualquier componente del sistema que requiera config y project_root al nacer."""
    def __init__(self, config: Dict[str, Any], project_root: str) -> None: ...