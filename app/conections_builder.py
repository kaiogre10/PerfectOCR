from typing import List, Any, Dict, Tuple
import logging
import ctypes

logger = logging.getLogger(__name__)

class ConectorsBuilder:
    """Builder que maneja y crea"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.export_targets = self.config["destination_services"]

    @property   
    def active_services(self) -> bool:
        return False if not self.export_targets else True

    def set_up_connectors(self, final_results: List[Tuple[int, int]]):
        for i, _ in enumerate(final_results):
            ptr, buff_size = final_results[i]
            # try:
            bytes_leidos = ctypes.string_at(ptr, buff_size)
                # raise MemoryError("Error leyendo bytecode")
            # except MemoryError as e:
            #     logger.warning(f"Error leyendo bytecode: {e}", exc_info=True)
            logger.info(f"BYTES_ALMACENADOS: '{bytes_leidos}'")
        return None
