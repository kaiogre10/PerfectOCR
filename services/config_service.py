from typing import Dict, Any, List
import logging
from config.config_validator import ConfigBuilder
from types import MappingProxyType

logger = logging.getLogger(__name__)

class ConfigService:
    __slots__ = ["validated_config"]
    """Gestor de los parametros de configuración"""
    def __init__(self, config_path: str, TEST_MODE: bool):
        self.validated_config = ConfigBuilder(config_path, TEST_MODE)

    @property
    def no_modules(self) -> bool:
        return self.validated_config.activate_modules

    @property
    def system_config(self):
        return self.validated_config.system_config

    @property
    def enabled_outputs(self) -> Dict[str, Any]:
        return self.validated_config.enabled_outputs

    @property
    def workers_order(self) -> Dict[str, List[str]]:
        return self.validated_config.workers_order

    @property
    def exporting_config(self) -> Dict[str, Any]:
        return self.validated_config.exporting_config

    @property
    def logs_debug(self) -> Dict[str, Any]:
        return self.validated_config.logs_debug

    @property
    def models_config(self) -> MappingProxyType:
        return self.validated_config.models_config

    @property
    def modules_config(self) -> Dict[str, Any]:
        return self.validated_config.modules_config

    @property
    def utils_config(self) -> Dict[str, Any]:
        return self.validated_config.utils_config

    @property
    def manager_config(self) -> Dict[str, MappingProxyType]:
        """Devuelve el paquete estándar de configuraciones de los managers"""
        return self.validated_config.manager_config
