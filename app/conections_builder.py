from typing import List, Any, Dict, Tuple, Optional
from core.workers.shipping.connectors_factory import ConnectorWorkersFactory
import logging
import ctypes
from app.distribution_manager import DistributionManager

logger = logging.getLogger(__name__)

class ConectorsBuilder:
    """Builder que maneja y crea"""
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.config = config

    @property   
    def active_services(self) -> bool:
        return False if not self.config["destination_services"] else True
    
    def create_conectors(self, context: Optional[Dict[str, Any]] = None):
        """Crea stager de preparación de imagen con configuraciones específicas del master config."""
        export_targets = self.config["destination_services"]
        if not export_targets:
            return None

        factory = self.get_conectors_factory(export_targets)
        if factory is None:
            return None

        conector_workers = factory.create_workers(export_targets, context)

        return DistributionManager(self.config, conector_workers, self.project_root)
    
    def get_conectors_factory(self, exporting_config: Dict[str, Any]) -> Optional[ConnectorWorkersFactory]:
        return None if not exporting_config else ConnectorWorkersFactory(exporting_config, self.project_root)

    def set_up_connectors(self, final_results: List[Tuple[int, int]]):
        
        for i, _ in enumerate(final_results):
            ptr, buff_size = final_results[i]
            # try:
            bytes_leidos = ctypes.string_at(ptr, buff_size)
                # raise MemoryError("Error leyendo bytecode")
            # except MemoryError as e:
            #     logger.warning(f"Error leyendo bytecode: {e}", exc_info=True)
            # logger.info(f"BYTES_ALMACENADOS: '{bytes_leidos}'")
        return None
