from typing import Any, List, Dict
import logging
from config.config_validator import ConfigBuilder
from types import MappingProxyType

logger = logging.getLogger(__name__)

class ConfigService:
    __slots__ = ["validated_config"]
    """Gestor de los parametros de configuración"""
    def __init__(self, config_path: str):
        self.validated_config = ConfigBuilder(config_path)

    @property
    def logs_debug(self) -> MappingProxyType[str, Any]:
        """Devuelve la configuración para los logs en termial y archivo.log"""
        return MappingProxyType(self.validated_config.logs_debug)

    @property
    def no_modules(self) -> bool:
        return self.validated_config.activate_modules

    @property
    def system_config(self) -> MappingProxyType[str, List[str]]:
        """Devuelve la configuración general del sistema"""
        return MappingProxyType(self.validated_config.system_config)

    @property
    def enabled_outputs(self) -> MappingProxyType[str, Dict[str, bool]]:
        """Devuelve la configuración de los outputs de debug visuales"""
        return MappingProxyType(self.validated_config.enabled_outputs)

    @property
    def workers_order(self) -> MappingProxyType[str, List[str]]:
        """Devuelve la configuración para la orquestación de los workers de procesamiento"""
        return MappingProxyType(self.validated_config.workers_order)

    @property
    def exporting_config(self) -> MappingProxyType[str, Any]:
        """Devuelve la configuración del modulo de exportación de datos"""
        return MappingProxyType(self.validated_config.exporting_config)

    @property
    def models_config(self) -> MappingProxyType[str, Any]:
        """Devuelve la configuración de los modelos entrenados"""
        return MappingProxyType(self.validated_config.models_config)

    @property
    def modules_config(self) -> MappingProxyType[str, Any]:
        """Devuelve la configuración de los modulos de procesamiento"""
        return MappingProxyType(self.validated_config.modules_config)

    @property
    def utils_config(self) -> MappingProxyType[str, Any]:
        """Devuelve la configuración generales"""
        return MappingProxyType(self.validated_config.utils_config)

    @property
    def manager_config(self) -> Dict[str, MappingProxyType[str, Any]]:
        """Devuelve el paquete estándar de configuraciones de los managers"""
        return self.validated_config.manager_config
    
    # Configuración optimizada para desarrollo ajustada a mi Pc (LATITUDE 5591, 32GB DE RAM, INTEL i5 H8400 DE 4 NÚCLEOS)
    @property
    def env_config(self) -> Dict[str, Any]:
        return self.validated_config.env_config