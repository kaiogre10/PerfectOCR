# services/config_loader.py
import yaml
import commentjson # type: ignore
import os
from typing import Any, List
from domain.config_models import MasterConfig
import services.log_service as log_service
import pickle

def load_config_file(default_config_path: List[str]):
    """Carga archivos de configuración, asegura agnosismo si cambia el formato de archivo de entrada"""
    system_config_file = os.path.join(*default_config_path, ("master_config" + ".yaml"))
    user_config_file = os.path.join(*default_config_path, ("user_config" + ".jsonc")) # Con comentarios temporalmente para flexibilidad

    config_json = _load_user_config(user_config_file)
    config_yaml = _load_yaml(system_config_file)
    config_yaml.update(config_json)    
    return MasterConfig.model_validate(obj=config_yaml, strict=True, extra='forbid', from_attributes=True, by_name=True).model_dump()

def _load_yaml(system_config_path: str):
    if not os.path.isfile(system_config_path):
        raise FileNotFoundError(f"ARCHIVO DE CONFIGURACIÓN NO ENCONTRADO: {system_config_path}")
    with open(system_config_path, 'r', encoding='utf-8') as f:
        system_raw = yaml.safe_load(f)
        if not system_raw or not isinstance(system_raw, dict):
            raise TypeError(f"DEAFULT CONFIG debe ser Dict[str, params], recibido: {type(system_raw).__name__}")
        return system_raw

def _load_user_config(user_config_file: str):
    if not os.path.isfile(user_config_file):
        raise FileNotFoundError(f"ARCHIVO DE CONFIGURACIÓN NO ENCONTRADO: {user_config_file}")
    with open(user_config_file, 'r', encoding='utf-8') as f:
        user_config_raw = commentjson.load(f) # type: ignore
        if not user_config_raw or not isinstance(user_config_raw, dict):
            raise TypeError(f"DEAFULT CONFIG debe ser Dict[str, params], recibido: {type(user_config_raw).__name__}")
        return user_config_raw
    
def load_pickle(pkl_path: str, mode: str):
    """Carga pickle"""
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Pickle no encontrado en {pkl_path}")
    with open(pkl_path, mode) as f:
        model_pkl = pickle.load(f)
        if not model_pkl:
            raise pickle.UnpicklingError("ERROR EN LA CARGA DEL PICKLE")
    model_pkl["allow_edit"] = False
    return model_pkl

def save_pickle(model_pkl: Any, pkl_path: str, mode: str):
    model_pkl["allow_edit"] = False
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Ruta inválida para guardar pickle: {pkl_path}")
    with open(pkl_path, mode) as f:
        pickle.dump(model_pkl, f, protocol=5)