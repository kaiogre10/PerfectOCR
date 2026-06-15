# services/config_service.py
import yaml
from typing import Dict, Any, cast, List, Set, Tuple, FrozenSet
from types import MappingProxyType
from functools import cached_property
from config.config_models import MasterConfig
import services.logs_service as log_service
import logging

logger = logging.getLogger(__name__)

elemental_worker = "image_loader"
det = "geometry_detector"
ocr_workers: Set[str] = set(["polygon_extractor", "paddle_wrapper", det])
full_ocr: FrozenSet[str] = frozenset(ocr_workers.union(["data_finder"]))
min_workers: FrozenSet[str] = frozenset(ocr_workers.union(set([elemental_worker]))) # "lineal", "vectorizer", "cos_sim", "table_structurer", "math_max", "data_collector"

class ConfigBuilder:
    """Validada de los parametros de configuración"""
    def __init__(self, config_path: str):
        self.config = self._load_and_validate_yaml(config_path)
        elemental_params = elemental_worker in self.create_stager[0][1]
        self.active_full_ocr = ocr_workers.issubset(self.all_workers)
        if not self._validate_config(elemental_params):
            self.config = {}
        else:
            self.config = self.config

    @cached_property
    def testing_modes(self) -> MappingProxyType[str, bool]:
        return MappingProxyType(self.config.get("test_modes",{}))

    @property
    def test_mode(self):
        return self.testing_modes.get("deploy_mode")

    @property
    def test_config(self):
        return self.testing_modes.get("test_config")
    
    @cached_property
    def no_activate_modules(self) -> bool:
        return (self.test_mode == True) and ((elemental_worker in self.create_stager[0][1]) == False)

    @cached_property
    def system_config(self) -> MappingProxyType[str, List[str]]:
        sys_config = self.config.get("system_config", {})
        return MappingProxyType({}) if not sys_config["input_dirs"] else MappingProxyType(sys_config)

    @cached_property
    def enabled_outputs(self) -> Dict[str, Dict[str, bool]]:
        if self.no_activate_modules or not self.system_config["output_paths"]:
            return {}
        return self.config.get("enabled_outputs", {})

    @cached_property
    def workers_order(self) -> Dict[str, List[str]]:
        return self.config.get("pipeline_secuence", {})

    @cached_property
    def exporting_config(self) -> Dict[str, List[str]]:
        return self.config.get("exporting_config", {})

    @cached_property
    def logs_debug(self) -> Dict[str, Any]:
        logs = self.config.get("log_debug", {})
        if self.no_activate_modules:
            return {}

        # Mutación controlada: Se ejecuta UNA SOLA VEZ y se queda en caché
        if logs.get("all_logs") or self.test_config:
            for key, value in logs.items():
                if isinstance(value, bool):
                    logs[key] = True
                elif isinstance(value, list):
                    logs[key] = [-1]
            return logs
        return logs
        
    @cached_property
    def models_config(self) -> Dict[str, Any]:
        if self.no_activate_modules:
            return {
                "models_config": self.config.get("models_config", {}),
                "activate_wf": True,
                "activate_rec": True,
                "activate_det": True
            }

        if not self.all_workers:
            return {}

        if det not in self.all_workers:
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

        if self.active_full_ocr:
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
    def modules_config(self) -> MappingProxyType[str, Any]:
        return MappingProxyType({}) if self.no_activate_modules else MappingProxyType(self.config.get("modules", {}))

    @cached_property
    def img_workers_config(self):
        return MappingProxyType({
            **self.modules_config.get("image_preparation", {}),
            **self.enabled_outputs.get("image_load_outputs", {})
        })

    @cached_property
    def img_prep_config(self) -> MappingProxyType[str, Any]:
        if self.no_activate_modules or not self.create_stager[0][1]:
            return MappingProxyType({})
        else:
            return MappingProxyType({
                "worker_config": self.img_workers_config,
                "imagepre_stage": self.workers_order["imagepre_stage"]
            })

    # @property
    # def preprocessing_workers_config(self):
    #     return MappingProxyType({
    #         **self.modules_config.get("image_preparation", {}),
    #         **self.enabled_outputs.get("image_load_outputs", {})
    #     })
    @cached_property
    def preprocessing_config(self)-> MappingProxyType[str, Any]:
        if self.no_activate_modules or not self.create_stager[1][1]:
            return MappingProxyType({})
        else:
            return MappingProxyType({
                **self.modules_config.get("preprocessing", {}),
                **self.enabled_outputs.get("preprocessing_outputs", {}),
                "preprocessing_stage": self.workers_order["preprocessing_stage"]
            })

    @cached_property
    def ocr_config(self) -> MappingProxyType[str, Any]:
        if self.no_activate_modules or not self.create_stager[2][1] or not self.active_full_ocr:
            return MappingProxyType({})
        else:
            create_refiners = self.modules_config.get("ocr", {}).get("text_refiner", {}).get("num_passes")
            return MappingProxyType({
                **self.modules_config.get("ocr", {}),
                **self.enabled_outputs.get("ocr_outputs", {}),
                **self.logs_debug,
                "ocr_stage": self.workers_order["ocr_stage"],
                "create_refiners": create_refiners > 0
            })
       
    @cached_property
    def vectorization_config(self) -> MappingProxyType[str, Any]:
        vect_stage = self.create_stager[3][1]
        if self.no_activate_modules or not vect_stage or not "lineal" in vect_stage or not self.active_full_ocr:
            return MappingProxyType({})
        else:
            return MappingProxyType({
                **self.modules_config.get("vectorization", {}),
                **self.enabled_outputs.get("vectorization_outputs", {}),
                "vector_stage": self.workers_order["vector_stage"]
            })

    @property
    def env_config(self) -> Dict[str, Any]:
        return self.config.get("env_config", {})
    
    @cached_property
    def create_stager(self) -> List[Tuple[str, List[str]]]:
        full_stage: List[Tuple[str, List[str]]] = []
        for stage_workers in self.workers_order.items():
            full_stage.append(stage_workers)
        return full_stage
    
    @cached_property
    def all_workers(self) -> FrozenSet[str]:
        img_prep = self.create_stager[0][1]
        all_workers: Set[str] = set(img_prep)

        if self.create_stager[1][1]:
            prep = self.create_stager[1][1]
            all_workers.update(prep)

        if self.create_stager[2][1]:
            ocr = self.create_stager[2][1]
            all_workers.update(ocr)

        if self.create_stager[3][1]:
            vect = self.create_stager[3][1]
            all_workers.update(vect)
        return frozenset(all_workers)
    
    def _validate_config(self, elemental_params: bool) -> bool:
        msg: str = f"MODO: {"TEST" if self.test_mode else "PRODUCIÓN"}, verificaciones_robustas: '{self.test_mode}'."
        if not self.system_config:
            log_service.log_active_areas("No hay rutas input")
            return False

        elif not self.test_mode and not elemental_params:
            log_service.log_active_areas(f"ERROR CRÍTICO, NO HAY IMAGE LOADER PARA PRODUCCIÓN")
            return False

        elif self.test_mode and not elemental_params:
            log_service.log_active_areas(msg, self.create_stager) # type: ignore
            return True

        elif self.test_mode:
            log_service.log_active_areas((msg + " Modulos:"), self.create_stager) # type: ignore
            return True

        elif not self.no_activate_modules and self._validate_min_workers():
            log_service.log_active_areas((msg + "Stages Activas:"), self.create_stager) # type: ignore
            return True
        else:
            log_service.log_active_areas(f"Error de configuración")
            return False

    def _load_and_validate_yaml(self, config_path: str) -> Dict[str, Any]:
        """Carga YAML y valida con Pydantic - ROBUSTEZ."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                raw = yaml.safe_load(f)
            if not isinstance(raw, dict):
                raise TypeError(f"Config raíz debe ser dict, recibido: {type(raw).__name__}")
            typed_raw = cast(Dict[str, Any], raw)
            return MasterConfig.model_validate(typed_raw).model_dump()

        except Exception as e:
            logger.error(f"Error validando configuración desde {config_path}: {e}", exc_info=True)
        return {}

    def _validate_min_workers(self) -> bool:
        try:
            if not self.workers_order:
                log_service.log_active_areas("ERROR: No hay configuración de workers disponible")
                return False
            elif min_workers.issubset(self.all_workers):
                return True
            else:
                workers_missing = min_workers - self.all_workers
                log_service.log_active_areas(f"Faltan: {workers_missing} de los '{len(min_workers)}' workers mínimos para el pipeline")
                return False
        except Exception as e:
            logger.error(f"Error crítico en la revisión de parámetros mínimos: {e}", exc_info=True)
        return False