# services/config_loader.py
import os
from typing import List
from domain.config_models import SystemSetUp, ConfigParams
from utils.file_handler import load_yaml, load_jsoncomment

def load_config(default_config_path: List[str]):
    """
    Carga archivos de configuración, asegura agnosismo si cambia el formato de archivo de entrada.
    return ['system_set_up', 'config_params']
    """
    config_params_file = os.path.join(*default_config_path, ("config_params" + ".yaml"))
    system_config_file = os.path.join(*default_config_path, ("master_config" + ".yaml"))
    user_config_file = os.path.join(*default_config_path, ("user_config" + ".jsonc")) # Con comentarios temporalmente para flexibilidad

    config_json = load_jsoncomment(user_config_file, 'r')
    if not isinstance(config_json, dict):
        raise TypeError(f"JSON CONFIG debe ser Dict[str, params], recibido: {type(config_json).__name__}")
    
    pams_yaml = load_yaml(config_params_file, 'r')
    setup_yaml = load_yaml(system_config_file, 'r')

    if not isinstance(setup_yaml, dict) or not isinstance(pams_yaml, dict):
        raise TypeError(f"YAML CONFIG debe ser Dict[str, params], recibido: {type(setup_yaml).__name__}") # type: ignore
    
    pams_yaml.update(config_json) # type: ignore
    config_params = ConfigParams.model_validate(obj=pams_yaml, strict=True, extra='forbid', from_attributes=True, by_name=True).model_dump()
    system_set_up = SystemSetUp.model_validate(obj=setup_yaml, strict=True, extra='forbid', from_attributes=True, by_name=True).model_dump()
    return system_set_up, config_params