# services/config_service.py
import yaml
from typing import Dict, Any, cast, List, Set, Tuple
from functools import cached_property
from config.config_models import MasterConfig
import logging

logger = logging.getLogger(__name__)

class ConfigService:
    """Gestor de los parametros de configuración"""
    def __init__(self, config_path: str, TEST_MODE: bool, output_paths: List[str]):
        validated_config = self._load_and_validate_yaml(config_path)
        self.config = validated_config.model_dump()
        elemental_worker: Set[str] = {"image_loader"}
        elemental_params = elemental_worker.issubset(self.create_stager[0][1])
        self.det = {"geometry_detector"}
        self.ocr_workers: Set[str] = {"polygon_extractor", "paddle_wrapper"}
        self.ocr_workers.update(self.det)
        self.min_workers: Set[str] = self.ocr_workers.union(elemental_worker) #"lineal", "vectorizer", "cos_sim", "table_structurer"
        self.no_modules = (elemental_params is False) and (TEST_MODE is True)
        self.enable_outputs = True if output_paths else False
        
        if not TEST_MODE and not elemental_params:
            logger.error(f"ERROR CRÍTICO, NO HAY IMAGE LOADER PARA PRODUCCIÓN")
            self.config = {}

        elif TEST_MODE and not elemental_params:
            logger.warning(f"TEST MODE ACTIVADO, verificaciones robustas desactivadas. '{self.log_active_areas()}'")
            self.config = self.config
            
        elif TEST_MODE:
            logger.warning(f"TEST MODE ACTIVADO, verificaciones robustas desactivadas. Modulos '{self.log_active_areas()}'")
            self.config = self.config

        elif not self.no_modules and self._validate_min_workers():
            logger.warning(f"MODO PRODUCCIÓN ACTIVADO, se realizarán validaciones robustas. Stages activas: '{self.log_active_areas()}'")
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
            logger.error(f"Error validando configuración desde {config_path}: {e}")
            raise

    @cached_property
    def activate_modules(self) -> bool:
        return self.no_modules
    
    @cached_property
    def enabled_outputs(self) -> Dict[str, Any]:
        if self.no_modules:
            logger.info("Enabled output desactivados, sin workers")
            return {}
            
        return self.config.get("enabled_outputs", {})

    @cached_property
    def workers_order(self) -> Dict[str, List[str]]:
        return self.config.get("pipeline_secuence", {})
        
    @cached_property
    def logs_debug(self) -> Dict[str, Any]:
        return {} if self.no_modules else self.config.get("log_debug", {})

    @cached_property
    def models_config(self) -> Dict[str, Any]:
        if self.no_modules:
            return {
                "models_config": self.config.get("models_config", {}),
                "activate_wf": True,
                "activate_rec": True,
                "activate_det": True
            }
            
        if not self.all_workers:
            logger.debug("Sin all workers")
            return {}

        finder: Set[str] = {"data_finder"}
        full_ocr = self.ocr_workers.union(finder)

        if not self.det.issubset(self.all_workers):
            logger.debug("Configuración: Sin geometry_detector, no se cargan modelos")
            return {}

        if full_ocr.issubset(self.all_workers):
            logger.debug("Configuración: OCR completo + Word Finder")
            return {
                "models_config": self.config.get("models_config", {}),
                "activate_wf": True,
                "activate_rec": True,
                "activate_det": True
            }

        if self.ocr_workers.issubset(self.all_workers):
            logger.debug("Configuración: OCR completo sin Word Finder")
            return {
                "models_config": self.config.get("models_config", {}),
                "activate_wf": False,
                "activate_rec": True,
                "activate_det": True
            }

        logger.debug("Configuración: Solo modelo de detección")
        return {
            "models_config": self.config.get("models_config", {}),
            "activate_wf": False,
            "activate_rec": False,
            "activate_det": True
        }
                
    @cached_property
    def modules_config(self) -> Dict[str, Any]:
        return {} if self.no_modules else self.config.get("modules", {})
    
    @cached_property
    def utils_config(self) -> Dict[str, Any]:
        return self.config.get("utils", {})
      
    @cached_property
    def img_prep_config(self) -> Dict[str, Any]:
        if self.no_modules:
            return {}
        else:
            return {
                **self.modules_config.get("image_preparation", {}),
                **self.enabled_outputs.get("image_load_outputs", {}),
                **self.utils_config,
                "imagepre_stage": self.workers_order["imagepre_stage"]
            }
       
    @cached_property
    def preprocessing_config(self)-> Dict[str, Any]:
        if self.no_modules or not self.create_stager[1][1]:
            return {}
        else:
            return {
                **self.modules_config.get("preprocessing", {}),
                **self.enabled_outputs.get("preprocessing_outputs", {}),
                **self.utils_config,
                "preprocessing_stage": self.workers_order["preprocessing_stage"]
            }

    @cached_property
    def ocr_config(self) -> Dict[str, Any]:
        if self.no_modules or not self.create_stager[2][1] or not self.ocr_workers.issubset(self.all_workers):
            return {}
        else:
            create_refiners = self.modules_config.get("ocr", {}).get("text_refiner", {}).get("num_passes")
            return {
                **self.modules_config.get("ocr", {}),
                **self.enabled_outputs.get("ocr_outputs", {}),
                **self.utils_config,
                "ocr_stage": self.workers_order["ocr_stage"],
                "create_refiners": create_refiners > 0,
                "semantic_types_log": self.logs_debug["semantic_types_log"],
                "seman_clas": self.logs_debug.get("seman_clas")
            }
       
    @cached_property
    def vectorization_config(self) -> Dict[str, Any]:
        vect_stage = self.create_stager[3][1]
        if self.no_modules or not vect_stage or not "lineal" in vect_stage or not self.ocr_workers.issubset(self.all_workers):
            return {}
        else:
            return {
                **self.modules_config.get("vectorization", {}),
                **self.enabled_outputs.get("vectorization_outputs", {}),
                **self.utils_config,
                "vector_stage": self.workers_order["vector_stage"]
            }
        
    @cached_property
    def manager_config(self) -> Dict[str, Dict[str, Any]]:
        """Devuelve el paquete estándar de configuraciones de los managers"""
        return {
            "image_preparation": self.img_prep_config,
            "preprocessing": self.preprocessing_config,
            "ocr": self.ocr_config,
            "vectorization": self.vectorization_config
        }

    def _validate_min_workers(self) -> bool:
        try:
            if not self.workers_order:
                logger.critical("No hay configuración de workers disponible")
                return False
            elif self.min_workers.issubset(self.all_workers):
                return True
            else:
                workers_missing: Set[str] = self.min_workers.difference(self.all_workers)
                logger.critical(f"Faltan: {workers_missing} de los '{len(self.min_workers)}' workers mínimos para el pipeline")
                return False
        except Exception as e:
            logger.error(f"Error crítico en la revisión de parámetros mínimos: {e}", exc_info=True)
        return False
    
    @cached_property
    def create_stager(self) -> List[Tuple[str, List[str]]]:
        full_stage: List[Tuple[str, List[str]]] = []
        for stage_workers in self.workers_order.items():
            full_stage.append(stage_workers)
        return full_stage
    
    @cached_property
    def all_workers(self) -> Set[str]:
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
        
    def log_active_areas(self):
        stages_list: List[str] = []
        for stage, stager in self.manager_config.items():
            if not stager:
                continue
            stage = stage.replace("_", " ", 1).title()
            stages_list.append(stage)
            
        return ", ".join(stages_list) if stages_list else "SOLO BUILDERS"
