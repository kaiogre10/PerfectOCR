# services/config_loader.py
import os
from domain.config_models import SystemSetUp, ConfigParams
from utils.file_handler import load_yaml, load_jsoncomment

def load_config(PROJECT_ROOT: str):
    """
    Carga archivos de configuración, asegura agnosismo si cambia el formato de archivo de entrada.
    return ['system_set_up', 'config_params']
    """
    file_dir = os.path.dirname(os.path.abspath(__file__))
    current_dir = os.path.join(PROJECT_ROOT, os.path.relpath(file_dir, PROJECT_ROOT))
    config_params_file = os.path.join(current_dir, ("config_params" + ".yaml"))
    system_config_file = os.path.join(current_dir, ("master_config" + ".yaml"))
    user_config_file = os.path.join(current_dir, ("user_config" + ".jsonc")) # Con comentarios temporalmente para flexibilidad

    config_json = load_jsoncomment(user_config_file, 'r')
    if not isinstance(config_json, dict):
        raise TypeError(f"JSON CONFIG debe ser Dict[str, params], recibido: {type(config_json).__name__}")
    
    pams_yaml = load_yaml(config_params_file, 'r')
    setup_yaml = load_yaml(system_config_file, 'r')

    if not isinstance(setup_yaml, dict) or not isinstance(pams_yaml, dict):
        raise TypeError(f"YAML CONFIG debe ser Dict[str, params], recibido: {type(setup_yaml).__name__}") # type: ignore
    
    pams_yaml.update(config_json) # type: ignore
    config_params = ConfigParams.model_validate(obj=pams_yaml, strict=True, extra='forbid', from_attributes=True, by_name=True).model_dump()
    master_config = SystemSetUp.model_validate(obj=setup_yaml, strict=True, extra='forbid', from_attributes=True, by_name=True).model_dump()
    master_config.update(config_params)
    return master_config