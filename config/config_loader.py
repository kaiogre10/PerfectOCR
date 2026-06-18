# services/config_loader.py
import yaml
from typing import cast, Dict, Any
from config.config_models import MasterConfig
import services.logs_service as log_service

def load_config_file(config_path: str) -> Dict[str, Any]:
        """Carga archivo de de condfiguración, asegura agnosismo si cambia el formato de archivo de entrada"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                raw = yaml.safe_load(f)
            if not isinstance(raw, dict):
                raise TypeError(f"Config raíz debe ser dict, recibido: {type(raw).__name__}")
            typed_raw = cast(Dict[str, Any], raw)
            return MasterConfig.model_validate(typed_raw).model_dump()

        except ValueError as e:
            log_service.basic_logger(f"Error validando configuración desde {config_path}: {e}")
        return {}
