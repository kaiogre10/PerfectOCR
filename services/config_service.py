# services/config_service.py
import yaml
from typing import Dict, Any, cast, List, Set, Optional
from config.config_models import MasterConfig
import logging

logger = logging.getLogger(__name__)

class ConfigService:
    """Gestor de los parametros de configuración"""
    def __init__(self, config_path: str, TEST_MODE: bool):
        self.config_path = config_path
        self.validated_config = self._load_and_validate_yaml(config_path)
        self.config = self.validated_config.model_dump()
        self.test_mode = TEST_MODE
        if not self.test_mode and self._validate_min_workers():
            logger.warning("Modo de producción activado, se cargan configuraciones robustas")
            self.config = self.config

        elif self.test_mode and "image_loader" in self.create_stager[0][1]:
            logger.warning(f"Modo de debug, restricciones robustas desactivadas")
            self.config = self.config

        else:
            logger.error(f"Error de configuración")
            self.config = {}
                
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
    def workers_order(self) -> Dict[str, Optional[List[str]]]:
        return self.config.get("pipeline_secuence", {})

    @property
    def models_config(self) -> Dict[str, Any]:
        ocr_workers = {"geometry_detector", "paddle_wrapper"}
        all_workers = self.get_all_workers
        ocr_active = not ocr_workers.isdisjoint(all_workers)

        if not all_workers:
            logger.debug("Sin all workers")
            return {}

    # WF solo activo con OCR completo + data_finder
        elif ocr_workers.issubset(all_workers) and "data_finder" in all_workers:
            logger.debug("OCR completo + data_finder")
            return {
 "models_config": self.config.get("models_config", {}),
            "activate_wf": True
            }

    # OCR parcial (uno o ambos, pero sin data_finder suficiente)
        elif ocr_active:
            logger.debug("OCR activo sin condiciones completas para WF")
            return {
            "models_config": self.config.get("models_config", {}),
            "activate_wf": False
        }

    # Sin OCR (incluye solo data_finder)
        elif "data_finder" in all_workers:
            logger.debug("Solo data_finder")
            return {
            "models_config": {},
            "activate_wf": False
        }

        else:
            logger.debug("Configuración de modelos no cargada")
            return {}
        
    @property
    def modules_config(self) -> Dict[str, Any]:
        return self.config.get("modules", {})
    
    @property
    def utils_config(self) -> Dict[str, Any]:
        return self.modules_config.get("utils", {})
      
    @property
    def img_prep_config(self) -> Dict[str, Any]:
        return {
            **self.modules_config.get("image_preparation", {}),
            **self.enabled_outputs.get("image_load_outputs", {}),
            **self.utils_config,
            "imagepre_stage": self.workers_order["imagepre_stage"]
        }
       
    @property
    def preprocessing_config(self)-> Dict[str, Any]:
        if not self.create_stager[1][1]:
            return {}
        else:
            return {
                **self.modules_config.get("preprocessing", {}),
                **self.enabled_outputs.get("preprocessing_outputs", {}),
                **self.utils_config,
                "preprocessing_stage": self.workers_order["preprocessing_stage"]
            }

    @property
    def ocr_config(self) -> Dict[str, Any]:
        if not self.create_stager[2][1]:
            return {}
        else:
            return {
                **self.modules_config.get("ocr", {}),
                **self.enabled_outputs.get("ocr_outputs", {}),
                **self.utils_config,
                "ocr_stage": self.workers_order["ocr_stage"]
            }
       
    @property
    def vectorization_config(self) -> Dict[str, Any]:
        if not self.create_stager[3][1]:
            return {}
        else:
            return {
                **self.modules_config.get("vectorization", {}),
                **self.enabled_outputs.get("vectorization_outputs", {}),
                **self.utils_config,
                "vector_stage": self.workers_order["vector_stage"]
            }
        
    @property
    def manager_config(self) -> Dict[str, Any]:
        """Devuelve el paquete estándar de configuraciones de los managers"""
        return {
            "image_preparation": self.img_prep_config,
            "preprocessing": self.preprocessing_config,
            "ocr": self.ocr_config,
            "vectorization": self.vectorization_config
        }

    def _validate_min_workers(self) -> bool:
        min_workers: Set[str] = {"image_loader", "geometry_detector", "polygon_extractor", "paddle_wrapper"}
        try:
            if not self.workers_order:
                logger.error("No hay configuración de workers disponible")
                return False
            
            set_worker_config = self.get_all_workers
            if min_workers.issubset(set_worker_config):
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
                workers_missing = min_workers - set_worker_config
                logger.warning(f"Faltan: {workers_missing} de los '{len(min_workers)}' workers mínimos para el pipeline")
                return False

        except Exception as e:
            logger.error(f"Error crítico en la revisión de parámetros mínimos: {e}", exc_info=True)
        return False
    
    @property
    def create_stager(self) -> List[str]:
        full_stage: List[str] = []
        for stage_workers in self.workers_order.items():
            full_stage.append(stage_workers) # type: ignore
        return full_stage
    
    @property
    def get_all_workers(self) -> Set[str]:
        all_workers = set(self.create_stager[0][1])

        if self.create_stager[1][1]:
            prep = set(self.create_stager[1][1])
            all_workers.update(prep)

        if self.create_stager[2][1]:
            ocr = set(self.create_stager[2][1])
            all_workers.update(ocr)

        if self.create_stager[3][1]:
            vect = set(self.create_stager[3][1])
            all_workers.update(vect)

        return all_workers
        