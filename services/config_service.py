# services/config_service.py
import yaml
import logging
from typing import Dict, Any, cast, List, Set
from config.config_models import MasterConfig

logger = logging.getLogger(__name__)

class ConfigService:
    """Fragmentador centralizado con validación robusta y flexibilidad."""
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.validated_config = self._load_and_validate_yaml(config_path)
        self.config = self.validated_config.model_dump()
                
    def _load_and_validate_yaml(self, config_path: str) -> MasterConfig:
        """Carga YAML y valida con Pydantic - ROBUSTEZ."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                raw = yaml.safe_load(f)
            if not isinstance(raw, dict):
                raise TypeError(f"Config raíz debe ser dict, recibido: {type(raw).__name__}")
            typed_raw = cast(Dict[str, Any], raw)
            return MasterConfig.model_validate(typed_raw)
            
        except Exception as e:
            logger.error(f"Error validando configuración desde {self.config_path}: {e}")
            raise

    @property
    def paths_config(self) -> Dict[str, str]:
        """Obtiene todas las rutas del sistema."""
        return self.config.get('paths', "")

    @property
    def enabled_outputs(self) -> Dict[str, bool]:
        """Obtiene flags de salida habilitados."""
        return self.config.get('enabled_outputs', {})

    @property
    def processing_config(self) -> Dict[str, Any]:
        """Obtiene configuración de procesamiento."""
        return self.config.get('processing', {})
        
    @property
    def modules_config(self) -> Dict[str, Dict[str, Any]]:
        """Obtiene configuración de módulos."""
        return {
            "modules": self.config.get("modules", {}),
            "enabled_outputs": self.enabled_outputs,
        }

    @property
    def validated_modules_config(self):
        """Acceso directo al objeto Pydantic validado."""
        return self.validated_config.modules

    @property
    def output_path(self) -> str:
        """Devuelve la ruta de la carpeta de salida."""
        return self.paths_config.get('output_folder', "")
    
    @property
    def workers_order(self) -> Dict[str, List[str]]:
        return self.config.get("pipeline_secuence", {})

    @property
    def models_config(self) -> Dict[str, Any]:
        """Obtiene la configuración global para modelos ML"""
        return { 
            "models_config": self.config.get("models_config", {}),
            "ocr_stage": self.workers_order.get("ocr_stage", [])
        }

    @property
    def manager_config(self) -> Dict[str, Any]:
        """Devuelve el paquete estándar de configuraciones de los managers"""
        return {
            "output_folder": self.output_path,
            "enabled_outputs": self.enabled_outputs,
            'secuence': self.workers_order
        }

    def validate_pipeline_config(self) -> bool:
        min_workers: List[str] = ["image_loader", "geometry_detector", "polygon_extractor", "paddle_wrapper"]
        set_min_workers: Set[str] = set(min_workers)
        num_min_workers = len(set_min_workers)
        if not self.workers_order:
            logger.error("No hay configuración de workers disponible")
            return False

        try:
            set_worker_config: Set[str] = {worker for stage in self.workers_order.values() for worker in stage}
            if set_min_workers.issubset(set_worker_config):
                for stage, stage_workers in self.workers_order.items():
                    logger.info(f"Activos '({len(stage_workers)}, {set(stage_workers)})' workers para '{stage}'")
                return True
            else:
                workers_missing = set_min_workers - set_worker_config
                logger.warning(f"Faltan: {workers_missing} de los '{num_min_workers}' workers mínimos para el pipeline")
                return False

        except Exception as e:
            logger.error(f"Error crítico en la revisión de parámetros mínimos: {e}", exc_info=True)
            return False
