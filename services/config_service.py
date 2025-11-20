# services/config_service.py
import yaml
from typing import Dict, Any, cast, List, Set, Optional
from config.config_models import MasterConfig
import logging

logger = logging.getLogger(__name__)

class ConfigService:
    """Gestor de los parametros de configuración"""
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.validated_config = self._load_and_validate_yaml(config_path)
        self.config = self.validated_config.model_dump()
        self.min_workers: List[str] = ["image_loader"]#, "geometry_detector", "polygon_extractor", "paddle_wrapper"]
                
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
    def processing_config(self) -> Dict[str, Any]:
        """Obtiene configuración de procesamiento."""
        return self.config.get('processing', {})

    @property
    def enabled_outputs(self) -> Dict[str, Any]:
        return self.config.get("enabled_outputs", {})
    
    @property
    def validated_modules_config(self):
        """Acceso directo al objeto Pydantic validado."""
        return self.validated_config.modules
    
    @property
    def workers_order(self) -> Dict[str, Optional[List[str]]]:
        return self.config.get("pipeline_secuence", {})

    @property
    def models_config(self) -> Dict[str, Any]:
        """Obtiene la configuración global para modelos ML"""
        return { 
            "models_config": self.config.get("models_config", {}),
            "ocr_stage": self.workers_order["ocr_stage"]
        }

    @property
    def models_config_validated(self) -> Dict[str, Any]:
        model_workers = ["geometry_detector", "paddle_wrapper"]
        if not self.validate_min_workers(model_workers):
            return {}
        else:
            return self.models_config
        
    @property
    def modules_config(self) -> Dict[str, Any]:
        return self.config.get("modules", {})
      
    @property
    def img_prep_config(self) -> Dict[str, Any]:
        if "imagepre_stage" in self.create_stager:
            return {
                **self.modules_config.get("image_preparation", {}),
                **self.enabled_outputs.get("image_load_outputs", {})
            }
        
        else:
            return {}
       
    @property
    def preprocessing_config(self)-> Dict[str, Any]:
        if "preprocessing_stage" in self.create_stager:
            return {
                **self.modules_config.get("preprocessing", {}),
                **self.enabled_outputs.get("preprocessing_outputs", {})
            }
        
        else:
            return {}

    @property
    def ocr_config(self) -> Dict[str, Any]:
        if "ocr_stage" in self.create_stager:
            return {
                **self.modules_config.get("ocr", {}),
                **self.enabled_outputs.get("ocr_outputs", {})
            }

        else:
            return {}
       
    @property
    def vectorization_config(self) -> Dict[str, Any]:
        if "vector_stage" in self.create_stager:
            return {
                **self.modules_config.get("vectorization", {}),
                **self.enabled_outputs.get("vectorization_outputs", {})
            }
        
        else:
            return {}
        
    @property
    def valid_modules_config(self) -> Dict[str, Any]:
        return {
            "image_preparation": self.img_prep_config,
            "preprocessing": self.preprocessing_config,
            "ocr": self.ocr_config,
            "vectorization": self.vectorization_config
        }
    
    @property
    def manager_config(self) -> Dict[str, Any]:
        """Devuelve el paquete estándar de configuraciones de los managers"""
        return {
            "stage_secuence": self.workers_order,
            "modules_config": self.valid_modules_config, 
        }

    def validate_min_workers(self, workers: List[str]) -> bool:
        if not workers:
            workers = self.min_workers

        self.set_min_workers: Set[str] = set(workers)
        
        if not self.workers_order:
            logger.error("No hay configuración de workers disponible")
            return False

        try:
            set_worker_config: Set[str] = set()
            # Construir conjunto de workers sólo desde stages válidos
            empty_stage: List[str] = []
            for stage, stage_workers in self.workers_order.items():
                if not stage_workers:
                    empty_stage.append(stage)
                    logger.debug(f"Stage '{stage}' sin workers, se ignora")
                    continue

                if isinstance(stage_workers, (list, tuple, set)): #type: ignore
                    # añadir sólo elementos tipo str
                    set_worker_config.update({w for w in stage_workers if isinstance(w, str)}) #type: ignore
                elif isinstance(stage_workers, str): #type: ignore
                    set_worker_config.add(stage_workers)
                else:
                    logger.debug(f"Stage '{stage}' con tipo inesperado {type(stage_workers).__name__}, se ignora")

            if self.set_min_workers.issubset(set_worker_config):
                # Loguear conteo por stage de forma segura
                for stage, stage_workers in self.workers_order.items():
                    if isinstance(stage_workers, (list, tuple, set)): #type: ignore
                        count = len(stage_workers)
                        workers_set = set(w for w in stage_workers if isinstance(w, str)) #type: ignore
                    elif isinstance(stage_workers, str): #type: ignore
                        count = 1
                        workers_set = {stage_workers}
                    else:
                        count = 0
                        workers_set: Set[str] = set()
                    logger.debug(f"Activos '({count}, {workers_set})' workers para '{stage}'")
                return True
            else:
                workers_missing = self.set_min_workers - set_worker_config
                logger.warning(f"Faltan: {workers_missing} de los '{len(self.set_min_workers)}' workers mínimos para el pipeline")
                return False

        except Exception as e:
            logger.error(f"Error crítico en la revisión de parámetros mínimos: {e}", exc_info=True)
            return False

    @property
    def create_stager(self) -> List[str]:
        active_stages: List[str] = []
        for stage, stage_workers in self.workers_order.items():
            if stage_workers:
                active_stages.append(stage)
            else:
                logger.debug(f"Stage '{stage}' sin workers, se ignora")

        return active_stages