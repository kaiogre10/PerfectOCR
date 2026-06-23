# services/config_validator.py
from typing import Dict, Any, List, Set, Tuple, FrozenSet
from types import MappingProxyType
from functools import cached_property
from services.log_service import log_active_areas, log_simple, basic_exc_logger
from config.config_loader import load_config_file

ELEMENTAL_WORKER = "image_loader"
det = "geometry_detector"
ocr_workers: Set[str] = set(["polygon_extractor", "paddle_wrapper", det])
full_ocr: FrozenSet[str] = frozenset(ocr_workers.union(["data_finder"]))
min_workers: FrozenSet[str] = frozenset(ocr_workers.union(set([ELEMENTAL_WORKER]))) # ["text_refiner", "lineal", "vectorizer", "cos_sim", "table_structurer", "math_max", "data_collector"]

class ConfigBuilder:
    """Valida de los parametros de configuración"""
    def __init__(self, config_path: List[str]):
        self.config = load_config_file(config_path)
        try:
            self._run_params
            self.active_full_ocr
        except Exception as e:
            log_simple(f"Error corriendo parametros: {e}")
            pass
        if not self._validate_config():
            self.config = {}
        else:
            self.config = self.config

    # Paramétros de sistema alto nivel
    @cached_property
    def elemental_params(self) -> bool:     # Condición mínima necesaria para que pueda arrancar el sistema
        return ELEMENTAL_WORKER in self.create_stager[0][1]

    @cached_property
    def active_full_ocr(self):
        return ocr_workers.issubset(self.all_workers)

    @cached_property
    def system_params(self):
        return self.config.get("system_params", {})

    @cached_property
    def system_paths(self) -> Dict[str, List[str]]:
        return self.system_params["system_paths"]

    @cached_property
    def deploy_settings(self) -> Dict[str, bool]:
        return self.config.get("deploy_settings", {})
    
    @cached_property
    def clean_project(self) -> bool:
        return bool(self.deploy_settings.get("clean_mode"))
    
    @cached_property
    def handle_memory(self) -> bool:
        return bool(self.deploy_settings.get("handle_memory"))
    
    @cached_property
    def deploy_mode(self) -> bool:
        return bool(self.deploy_settings.get("deploy_mode"))

    @cached_property
    def test_config(self) -> bool:
        return bool(self.deploy_settings.get("test_config"))
    
    @cached_property
    def db_local(self) -> bool:
        return bool(self.deploy_settings.get("postgre_local"))
    
    @cached_property
    def no_activate_modules(self) -> bool: # Parametro automátizado que permite arrancar el sistema para testear parametros de alto nivel sin crear objetos pesados de manera innecesaria
        return (self.deploy_mode == True) and ((self.elemental_params) == False)
    
    @cached_property
    def enabled_outputs(self) -> Dict[str, Dict[str, bool]]:
        if not self.system_paths["output_paths"] or self.no_activate_modules:
            return {}
        else:
            return self.config.get("enabled_outputs", {})
        
    @cached_property
    def user_requests(self):
        return self.config.get("user_requests", {})
    
    @cached_property
    def input_paths(self) -> Dict[str, List[str]]:
        user_dirs = self.user_requests.get("dirs", {})
        return {} if (not user_dirs["input_dirs"] and not self.deploy_mode) else user_dirs

    @cached_property
    def payload_request(self) -> Dict[str, Any]:
        return self.config.get("payload_request", {})

    @cached_property
    def workers_order(self) -> Dict[str, List[str]]:
        return self.config.get("pipeline_secuence", {})

    @cached_property
    def logs_debug(self) -> Dict[str, Any]:
        logs = self.config.get("log_debug", {})
        # Mutación controlada: Se ejecuta UNA SOLA VEZ y se queda en caché
        if logs.get("all_logs") or self.test_config:
            for key, value in logs.items():
                if isinstance(value, bool):
                    logs[key] = True
                elif isinstance(value, list):
                    logs[key] = [-1]
            logs.update({"handle_memory": self.handle_memory})
            return logs
        logs.update({"handle_memory": self.handle_memory})
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
    def modules_config(self) -> Dict[str, Any]:
        return {} if self.no_activate_modules else self.config.get("modules", {})

    @cached_property
    def img_prep_config(self) -> Tuple[Dict[str, Any], List[str]]:
        if self.no_activate_modules or not self.create_stager[0][1]:
            return ({}, [])
        else:
            image_workers = self.workers_order["image_preparation_stager"]
            if "polygon_extractor" in image_workers:
                if not det in image_workers:
                    self.workers_order["image_preparation_stager"].remove("polygon_extractor")

            config_module = self.modules_config.get("image_preparation", {})
            enabled_outputs = self.enabled_outputs.get("image_load_outputs", {})
            config_module.update(enabled_outputs)
            return config_module, self.workers_order["image_preparation_stager"]
            
    @cached_property
    def preprocessing_config(self)-> Tuple[Dict[str, Any], List[str]]:
        if self.no_activate_modules or not self.create_stager[1][1] or not self.active_full_ocr:
            return ({}, [])
        else:
            try:
                config_module = self.modules_config.get("preprocessing", {})
                enabled_outputs = self.enabled_outputs.get("preprocessing_outputs", {})
                config_module.update(enabled_outputs)
                return config_module, self.workers_order["preprocessing_stager"]
            except ValueError as e:
                basic_exc_logger(f"ERROR EN CDONFIGURACIÓN DE STAGER: {e}", exc_info=True)
            return ({}, [])
            
    @cached_property
    def ocr_config(self) -> Tuple[Dict[str, Any], List[str]]:
        if self.no_activate_modules or not self.create_stager[2][1] or not self.active_full_ocr:
            return ({}, [])
        else:
            config_module = self.modules_config.get("ocr", {})
            config_module.update(self.logs_debug)
            config_module.update(self.enabled_outputs.get("ocr_outputs"))
            return config_module, self.workers_order["ocr_stager"]
               
    @cached_property
    def vectorization_config(self) -> Tuple[Dict[str, Any], List[str]]:
        vect_stage = self.create_stager[3][1]
        if self.no_activate_modules or not vect_stage or not "lineal" in vect_stage or not self.active_full_ocr:
            return ({}, [])
        else:
            config_module = self.modules_config.get("vectorization", {})
            enabled_outputs = self.enabled_outputs.get("vectorization_outputs", {})
            config_module.update(enabled_outputs)
            config_module.update(self.payload_request)
            return config_module, self.workers_order["vectorization_stager"]
        
    @cached_property
    def local_db_config(self) -> MappingProxyType[str, Any]:
        if self.no_activate_modules or not self.db_local or not self.active_full_ocr:
            return MappingProxyType({})
        else:
            math_max_config = self.modules_config.get("vectorization", {})
            return MappingProxyType({
                **math_max_config.get("math_max", {}),
                "postgre_local": self.workers_order["db_stage"]
            })

    @cached_property
    def stagers_config(self) -> MappingProxyType[str, Tuple[Dict[str, Any], List[str]]]:
        """
        Se ejecuta UNA SOLA VEZ en el primer llamado de todo el pipeline.
        Construye la estructura y congela el mapa. Los subsecuentes accesos
        de los workers leerán directamente de la memoria RAM sin ejecutar código.
        """
        return MappingProxyType({
            "image_preparation_stager": self.img_prep_config,
            "preprocessing_stager": self.preprocessing_config,
            "ocr_stager":  self.ocr_config,
            "vectorization_stager": self.vectorization_config
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
    
    def _validate_config(self) -> bool:
        mode = (f'{"DEPLOY" if self.deploy_mode else "PRODUCTION"}') if not self.no_activate_modules else ("CONFIG TESTING" if self.test_config else "NO MODULES")
        msg: str = f"Modo: '{mode}', Validación: '{"COMPLETA" if not self.deploy_mode else "MÍNIMA"}' de la configuración."
        
        if not self.input_paths:
            log_simple("No hay rutas input")
            return False

        elif not self.deploy_mode and not self.elemental_params:
            log_simple(f"ERROR CRÍTICO, NO HAY IMAGE LOADER PARA PRODUCCIÓN")
            return False
        
        elif not self.deploy_mode and not self.handle_memory:
            log_simple(f"ACTIVAR MEMORIA DINÁMICA")
            return False

        elif self.no_activate_modules:
            log_active_areas(msg + " 'SOLO MODULOS DE ALTO NIVEL'") # type: ignore
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

    @cached_property
    def _run_params(self):
        self.elemental_params
        self.all_workers
        self.active_full_ocr
        self.workers_order
        self.stagers_config
        self.models_config
        self.logs_debug