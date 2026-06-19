# services/config_validator.py
from typing import Dict, Any, List, Set, Tuple, FrozenSet
from types import MappingProxyType
from functools import cached_property
from services.log_service import log_active_areas, log_simple
from config.config_loader import load_config_file

ELEMENTAL_WORKER = "image_loader"
det = "geometry_detector"
ocr_workers: Set[str] = set(["polygon_extractor", "paddle_wrapper", det])
full_ocr: FrozenSet[str] = frozenset(ocr_workers.union(["data_finder"]))
min_workers: FrozenSet[str] = frozenset(ocr_workers.union(set([ELEMENTAL_WORKER]))) # ["text_refiner", "lineal", "vectorizer", "cos_sim", "table_structurer", "math_max", "data_collector"]

class ConfigBuilder:
    """Validada de los parametros de configuración"""
    def __init__(self, config_path: str):
        self.config = load_config_file(config_path)
        self.active_full_ocr = ocr_workers.issubset(self.all_workers)
        if not self._validate_config():
            self.config = {}
        else:
            self.config = self.config

    @cached_property
    def elemental_params(self) -> bool:
        return ELEMENTAL_WORKER in self.create_stager[0][1]
    
    @cached_property
    def testing_modes(self) -> Dict[str, bool]:
        return self.config.get("test_modes",{})

    @property
    def deploy_mode(self) -> bool:
        return bool(self.testing_modes.get("deploy_mode"))

    @property
    def test_config(self) -> bool:
        return bool(self.testing_modes.get("test_config"))
    
    @cached_property
    def no_activate_modules(self) -> bool:
        return (self.deploy_mode == True) and ((self.elemental_params) == False)

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
            #logger.debug("Configuración: Sin geometry_detector, no se cargan modelos")
            return {}

        if full_ocr.issubset(self.all_workers):
            #logger.debug("Configuración: OCR completo + Word Finder")
            return {
                "models_config": self.config.get("models_config", {}),
                "activate_wf": True,
                "activate_rec": True,
                "activate_det": True
            }

        if self.active_full_ocr:
            #logger.debug("Configuración: OCR completo sin Word Finder")
            return {
                "models_config": self.config.get("models_config", {}),
                "activate_wf": False,
                "activate_rec": True,
                "activate_det": True
            }

        #logger.debug("Configuración: Solo modelo de detección")
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
    def img_prep_config(self) -> MappingProxyType[str, Any]:
        if self.no_activate_modules or not self.create_stager[0][1]:
            return MappingProxyType({})
        else:
            return MappingProxyType({
                **self.modules_config.get("image_preparation", {}),
                **self.enabled_outputs.get("image_load_outputs", {}),
                "image_preparation_stager": self.workers_order["image_preparation_stager"]
            })

    @cached_property
    def preprocessing_config(self)-> MappingProxyType[str, Any]:
        if self.no_activate_modules or not self.create_stager[1][1] or not self.active_full_ocr:
            return MappingProxyType({})
        else:
            return MappingProxyType({
                **self.modules_config.get("preprocessing", {}),
                **self.enabled_outputs.get("preprocessing_outputs", {}),
                "preprocessing_stager": self.workers_order["preprocessing_stager"]
            })

    @cached_property
    def ocr_config(self) -> MappingProxyType[str, Any]:
        if self.no_activate_modules or not self.create_stager[2][1] or not self.active_full_ocr:
            return MappingProxyType({})
        else:
            create_refiners = self.modules_config.get("ocr", {}).get("text_refine", {}).get("num_passes")
            return MappingProxyType({
                **self.modules_config.get("ocr", {}),
                **self.enabled_outputs.get("ocr_outputs", {}),
                **self.logs_debug,
                "ocr_stager": self.workers_order["ocr_stager"],
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
                "vectorization_stager": self.workers_order["vectorization_stager"]
            })
        
    @cached_property
    def local_db_config(self) -> MappingProxyType[str, Any]:
        db_stage = self.create_stager[4][1]
        if self.no_activate_modules or not db_stage or not self.active_full_ocr:
            return MappingProxyType({})
        else:
            math_max_config = self.modules_config.get("vectorization", {})
            return MappingProxyType({
                **math_max_config.get("math_max", {}),
                "postgre_local": self.workers_order["db_stage"]
            })

    @cached_property
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
    
    def _validate_config(self) -> bool:
        mode = (f'{"DEPLOY" if self.deploy_mode else "PRODUCTION"}') if not self.test_config else "CONFIG TESTING"
        msg: str = f"Modo: '{mode}', Validación: '{"MÍNIMA" if mode == "DEPLOY" else "COMPLETA"}' de la configuración."
        if not self.system_config:
            log_simple("No hay rutas input")
            return False

        elif not self.deploy_mode and not self.elemental_params:
            log_simple(f"ERROR CRÍTICO, NO HAY IMAGE LOADER PARA PRODUCCIÓN")
            return False

        elif self.deploy_mode and not self.elemental_params:
            log_active_areas(msg, self.create_stager) # type: ignore
            return True

        elif self.deploy_mode:
            log_active_areas((msg + " Modulos:"), self.create_stager) # type: ignore
            return True

        elif not self.no_activate_modules and self._validate_min_workers():
            log_active_areas((msg + "Stages Activas:"), self.create_stager) # type: ignore
            return True
        else:
            log_simple(f"Error de configuración")
            return False

    def _validate_min_workers(self) -> bool:
        try:
            if not self.workers_order:
                log_simple("ERROR: No hay configuración de workers disponible")
                return False
            elif min_workers.issubset(self.all_workers):
                return True
            else:
                workers_missing = min_workers - self.all_workers
                log_simple(f"Faltan: {workers_missing} de los '{len(min_workers)}' workers mínimos para el pipeline")
                return False
        except Exception as e:
            log_simple(f"Error crítico en la revisión de parámetros mínimos: {e}")
        return False