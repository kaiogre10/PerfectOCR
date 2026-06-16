from typing import Any, Dict, Optional
from core.workers.shipping.connectors_factory import ConnectorWorkersFactory
import logging
from services.gateaway_service import ServiceGateaway

logger = logging.getLogger(__name__)

class ConectorsBuilder:
    """Builder que maneja y crea"""
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.exporting_config = config
        dns = self.exporting_config.get("dns")
        self.gateaway_service = ServiceGateaway(dns)

    @property
    def active_services(self) -> bool:
        return False if not (self.exporting_config["destination_services"] and self.gateaway_service.test_local_connection()) else True
    
    def create_conectors(self, context: Optional[Dict[str, Any]] = None):
        """Crea stager de preparación de imagen con configuraciones específicas del master config."""
        exporting_config = self.exporting_config
        export_targets = exporting_config["destination_services"]
        if not export_targets:
            return None

        factory = self.get_conectors_factory(exporting_config)
        if factory is None:
            return None

        return factory.create_workers(export_targets, context)
    
    def get_conectors_factory(self, exporting_config: Dict[str, Any]) -> Optional[ConnectorWorkersFactory]:
        return None if not exporting_config else ConnectorWorkersFactory(exporting_config, self.project_root)

    # def set_up_connectors(self, final_results: List[Tuple[int, int]]):
        
    #     for i, _ in enumerate(final_results):
    #         ptr, buff_size = final_results[i]
    #         # try:
    #         bytes_leidos = ctypes.string_at(ptr, buff_size)
    #         # base_ptr, offsets = _request_storage(plano, BUUF_SIZES)
    #         # elemento i → base_ptr + offsets[i], longitud → offsets[i+1] - offsets[i]
    #     return None
