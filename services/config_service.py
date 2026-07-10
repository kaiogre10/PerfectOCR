from typing import Any, List, Dict, Tuple
from config.config_validator import ConfigValidator
from config.config_loader import load_config
# from domain.protocol import ConfigProtocol
# from typing import Dict, Any, List, Callable, TypeVar, Generic, Optional, ClassVar

# T = TypeVar('T', bound=ConfigProtocol)

class ConfigService:
    """API limpia que expone parametros de configuración al resto del Pipeline"""
    def __init__(self, config_path: List[str]):
        # self.validated_config = validated_config
        configs = load_config(config_path)
        setup = configs[0]
        params = configs[1]
        setup.update(params)
        self.validated_config = ConfigValidator(config_path[0], setup)

    @property
    def test_wf_model(self) -> bool:
        return self.validated_config.test_wf_model
    
    @property
    def test_config(self) -> bool:
        return self.validated_config.test_config

    @property
    def no_activate_modules(self) -> bool:
        return self.validated_config.no_activate_modules

    @property
    def handle_memory(self) -> bool:
        return self.validated_config.handle_memory

    @property
    def clean_project(self) -> bool:
        return self.validated_config.clean_project

    @property
    def logs_debug(self) -> Dict[str, Any]:
        """Devuelve la configuración para los logs en termial y archivo"""
        return self.validated_config.logs_debug

    @property
    def system_paths(self) -> Dict[str, List[str]]:
        """Devuelve las rutas internas de archivos relevantes del sistema"""
        return self.validated_config.system_paths

    @property
    def enabled_outputs(self) -> Dict[str, Dict[str, bool]]:
        """Devuelve la configuración de los outputs de debug visuales"""
        return self.validated_config.enabled_outputs

    @property
    def models_config(self) -> Dict[str, Any]:
        """Devuelve la configuración de los modelos entrenados"""
        return self.validated_config.models_config

    @property
    def stagers_config(self):
        return self.validated_config.stagers_config

    @property
    def create_stager(self) -> List[Tuple[str, List[str]]]:
        return self.validated_config.create_stager
    
    # Configuración optimizada para desarrollo ajustada a mi Pc (LATITUDE 5591, 32GB DE RAM, INTEL i5 H8400 DE 4 NÚCLEOS)
    @property
    def env_config(self) -> Dict[str, Any]:
        return self.validated_config.env_config

    @property
    def input_paths(self) -> Dict[str, List[str]]:
        return self.validated_config.input_paths