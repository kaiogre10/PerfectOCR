import logging
from typing import Optional, List, Dict, Any, Tuple
# import time
# import services.gateaway_service as gateaway_service

logger = logging.getLogger(__name__)

class DistributionManager:
    def __init__(self, config: Dict[str,Any], conectors: Any, project_root: str):
        self.config = config.get("exporting_config")
        self.project_root = project_root
        self.conectors = conectors

    # def distibute(self, payload_dirs: Tuple[int, int]):
        