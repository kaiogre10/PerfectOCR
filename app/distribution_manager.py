import logging
from typing import List, Dict, Any, Tuple
from core.factory.abstract_worker import ConnectorAbstractWorker
import time
from services.gateaway_service import ServiceGateaway

logger = logging.getLogger(__name__)

class DistributionManager:
    def __init__(self, config: Dict[str,Any], conectors: List[ConnectorAbstractWorker], project_root: str):
        self.config = config.get("exporting_config")
        self.project_root = project_root
        self.conectors = conectors

    def distibute(self, payload_dirs: List[Tuple[int, List[int]]]):
        start_time = time.perf_counter()
        gateaway_service = ServiceGateaway()
        exec_context: Dict[str, Any] = {}
        exec_context["payload_dirs"] = payload_dirs

        for conector_idx, conector in enumerate(self.conectors):
            worker_name = conector.__class__.__name__
            logger.info(f"Ejecutando {conector_idx + 1}/{len(self.conectors)}: '{worker_name}'")

            exec_context["worker_name"] = worker_name  # Actualiza el nombre en cada iteración
            
            worker_start = time.perf_counter()
            if not conector.transfer(exec_context, gateaway_service):
                worker_time = time.perf_counter() - start_time
                logger.error(f"'{worker_name}' falló, tiempo: {worker_start:.6f}'s", exc_info=True)
                return None, worker_time
        
        gateaway_service.stop_postgres()
        return time.perf_counter() - start_time