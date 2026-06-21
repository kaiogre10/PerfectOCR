# services/config_loader.py
import yaml
import commentjson # type: ignore
import os
from typing import cast, Dict, Any, List
from config.config_models import MasterConfig
import services.log_service as log_service

def load_config_file(default_config_path: List[str]) -> Dict[str, Any]:
    """Carga archivos de configuración, asegura agnosismo si cambia el formato de archivo de entrada"""
    system_config_file = os.path.join(*default_config_path, ("master_config" + ".yaml"))
    user_config_file = os.path.join(*default_config_path, ("user_config" + ".jsonc")) # Con comentarios temporalmente para flexibilidad

    config_json = _load_user_config(user_config_file)
    config_yaml = _load_yaml(system_config_file)
    config_yaml.update(config_json)    
    return MasterConfig.model_validate(config_yaml).model_dump()

def _load_yaml(system_config_path: str):
    try:
        with open(system_config_path, 'r', encoding='utf-8') as f:
            system_raw = yaml.safe_load(f)
    except FileNotFoundError as e:
        log_service.basic_exc_logger(f"ERROR BUSCANDO ARCHIVO DE CONFIGURACIÓN GLOBAL: {e}")
        raise
    try:
        if not isinstance(system_raw, dict):
            raise TypeError(f"DEAFULT CONFIG debe ser Dict[str, params], recibido: {type(system_raw).__name__}")
        return cast(Dict[str, Any], system_raw)
    except ValueError as e:
        log_service.basic_exc_logger(f"Error validando configuración desde {system_raw}: {e}", exc_info=True)
        raise

def _load_user_config(user_config_file: str):
    try:
        with open(user_config_file, 'r', encoding='utf-8') as f:
            user_config_raw = commentjson.load(f) # type: ignore
    except FileNotFoundError as e:
        log_service.basic_exc_logger(f"ERROR BUSCANDO ARCHIVO DE CONFIGURACIÓN PARA EL USUARIO: {e}")
        raise
    try:
        if not isinstance(user_config_raw, dict):
            raise TypeError(f"DEAFULT CONFIG debe ser Dict[str, params], recibido: {type(user_config_raw).__name__}")
        return cast(Dict[str, Any], user_config_raw)
    except ValueError as e:
        log_service.basic_exc_logger(f"Error validando configuración desde {user_config_raw}: {e}", exc_info=True)
        raise