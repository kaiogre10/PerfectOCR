from typing import Any, List, Dict, Tuple
import logging
from config.config_validator import ConfigBuilder
from types import MappingProxyType
from functools import cached_property

logger = logging.getLogger(__name__)

class ConfigService:
    """Gestor de los parametros de configuración"""
    __slots__ = (
        "validated_config", 
        "_manager_config", 
        "__dict__"
        )
    def __init__(self, config_path: str):
        self.validated_config = ConfigBuilder(config_path)

    @property
    def test_config(self):
        return self.validated_config.test_config

    @property
    def logs_debug(self) -> Dict[str, Any]:
        """Devuelve la configuración para los logs en termial y archivo"""
        return self.validated_config.logs_debug

    @property
    def no_activate_modules(self) -> bool:
        return self.validated_config.no_activate_modules

    @property
    def system_config(self) -> MappingProxyType[str, List[str]]:
        """Devuelve la configuración general del sistema"""
        return MappingProxyType(self.validated_config.system_config)

    @property
    def enabled_outputs(self) -> MappingProxyType[str, Dict[str, bool]]:
        """Devuelve la configuración de los outputs de debug visuales"""
        return MappingProxyType(self.validated_config.enabled_outputs)

    @property
    def exporting_config(self) -> MappingProxyType[str, Any]:
        """Devuelve la configuración del modulo de exportación de datos"""
        return MappingProxyType(self.validated_config.exporting_config)

    @property
    def models_config(self) -> MappingProxyType[str, Any]:
        """Devuelve la configuración de los modelos entrenados"""
        return MappingProxyType(self.validated_config.models_config)

    @cached_property
    def manager_config(self) -> MappingProxyType[str, Any]:
        """
        Se ejecuta UNA SOLA VEZ en el primer llamado de todo el pipeline.
        Construye la estructura y congela el mapa. Los subsecuentes accesos
        de los workers leerán directamente de la memoria RAM sin ejecutar código.
        """
        return MappingProxyType({
            "image_preparation_stager": self.validated_config.img_prep_config,
            "preprocessing_stager": self.validated_config.preprocessing_config,
            "ocr_stager": self.validated_config.ocr_config,
            "vectorization_stager": self.validated_config.vectorization_config
        })
    
    @property
    def local_db_config(self) -> MappingProxyType[str, Any]:
        return self.validated_config.local_db_config
    
    @property
    def create_stager(self) -> List[Tuple[str, List[str]]]:
        return self.validated_config.create_stager
    
    # Configuración optimizada para desarrollo ajustada a mi Pc (LATITUDE 5591, 32GB DE RAM, INTEL i5 H8400 DE 4 NÚCLEOS)
    @property
    def env_config(self) -> Dict[str, Any]:
        return self.validated_config.env_config