from functools import cached_property
from typing import Dict, Any, List, Set, Tuple, FrozenSet
from core.assets.assets import ELEMENTAL_WORKER, OCR_WORKERS, MIN_WORKERS
from services.log_service import log_simple
from config.config_loader import load_config
from config.config_logger import ConfigLogger

_elemental_workers = ELEMENTAL_WORKER
_ocr_workers = OCR_WORKERS     # Es parte de los min_workers pero por motivos de deploy lo mantendremos fuera
_min_workers = MIN_WORKERS

class BaseConfig:
    def __init__(self, project_root: str):
        self.config = load_config(project_root)

        if not self._validate_config():
            self.config = {}
            del self.config

    @cached_property
    def _elemental_params(self) -> bool:
        return _elemental_workers in self._create_stager[0][1]

    @cached_property
    def _active_full_ocr(self):
        return _ocr_workers.issubset(self._all_workers)

    @property
    def _system_params(self):
        return self.config.get("system_params", {})

    @property
    def _deploy_settings(self) -> Dict[str, bool]:
        return self.config.get("deploy_settings", {})

    @property
    def _clean_project(self) -> bool:
        return bool(self._deploy_settings.get("clean_mode"))

    @cached_property
    def _handle_memory(self) -> bool:
        return bool(self._deploy_settings.get("handle_memory"))

    @property
    def _deploy_mode(self) -> bool:
        return bool(self._deploy_settings.get("deploy_mode"))

    @property
    def _test_config(self) -> bool:
        return bool(self._deploy_settings.get("test_config"))

    @cached_property
    def _update_model(self) -> bool:
        return bool(self._deploy_settings.get("update_model"))

    @property
    def _log_config(self):
        return self.config.get("log_config", {})

    @cached_property
    def _test_wf_model(self) -> bool:
        test_wf_model = bool(self._deploy_settings.get("test_wf_model"))
        self._update_model = False if test_wf_model else self._update_model
        return test_wf_model

    @property
    def _db_local(self) -> bool:
        return bool(self._deploy_settings.get("postgre_local"))

    @property
    def _compile_cython(self) -> bool:
        return bool(self._deploy_settings.get("compile_cython"))

    @cached_property
    def _no_activate_modules(self) -> bool: # Parametro automátizado que permite arrancar el sistema para testear parametros de alto nivel sin crear objetos pesados de manera innecesaria
        """(deploy_mode == True) and (elemental_params == False)"""
        return (self._deploy_mode == True) and (self._elemental_params == False)

    @cached_property
    def _not_run_system(self) -> bool:
        """(self._deploy_mode == False) and (self._elemental_params == False)"""
        return (self._deploy_mode == False) and (self._elemental_params == False)

    @property
    def _enabled_outputs(self):
        return self.config.get("enabled_outputs", {})

    @property
    def _env_config(self) -> Dict[str, Any]:
        return self.config.get("env_config", {})

    @property
    def _user_requests(self) -> Dict[str, Dict[str, Any]]:
        return self.config.get("user_requests", {})

    @cached_property
    def _workers_order(self) -> Dict[str, List[str]]:
        return self.config.get("pipeline_secuence", {})

    @property
    def _modules_config(self):
        return self.config.get("modules", {})

    @cached_property
    def _input_paths(self):
        _input_paths = self._user_requests.get("input_paths", {})
        _input_paths["images_names"] = set(_input_paths["images_names"])
        _input_paths["skip_names"] = set(_input_paths["skip_names"])
        return _input_paths

    @cached_property
    def input_paths(self) -> Dict[str, List[str]]:
        input_paths = self._input_paths
        return {} if (not input_paths["input_dirs"] and not self._deploy_mode) else input_paths

    @cached_property
    def _models_config(self) -> Dict[str, Any]:
        return self.config.get("models_config", {})

    @cached_property
    def _create_stager(self) -> List[Tuple[str, List[str]]]:
        full_stage: List[Tuple[str, List[str]]] = []
        for stage_workers in self._workers_order.items():
            full_stage.append(stage_workers)
        return full_stage

    @cached_property
    def _all_workers(self) -> FrozenSet[str]:
        img_prep = self._create_stager[0][1]
        all_workers: Set[str] = set(img_prep)

        if self._create_stager[1][1]:
            prep = self._create_stager[1][1]
            all_workers.update(prep)

        if self._create_stager[2][1]:
            ocr = self._create_stager[2][1]
            all_workers.update(ocr)

        if self._create_stager[3][1]:
            vect = self._create_stager[3][1]
            all_workers.update(vect)
        return frozenset(all_workers)

    def _validate_config(self) -> bool:
        mode = f'{"DEPLOY" if self._deploy_mode else "PRODUCTION"}' if not self._no_activate_modules else ("CONFIG TESTING" if self._test_config else "NO MODULES")
        msg: str = f"Modo: '{mode}', Validación: '{"COMPLETA" if not self._deploy_mode else "MÍNIMA"}' de la configuración."

        if not self.input_paths:
            log_simple("No hay rutas input")
            return False

        elif self._not_run_system:
            log_simple("ERROR CRÍTICO, NO HAY IMAGE LOADER PARA PRODUCCIÓN")
            return False

        elif self._no_activate_modules:
            ConfigLogger.log_active_areas(msg + " 'SOLO MODULOS DE ALTO NIVEL'")
            return True

        elif not self._deploy_mode and not self._handle_memory:
            log_simple("ACTIVAR MEMORIA DINÁMICA")
            return False

        elif self._deploy_mode:
            ConfigLogger.log_active_areas((msg + " Modulos:"), self._create_stager)
            return True

        elif not self._no_activate_modules and self._validate_min_workers():
            ConfigLogger.log_active_areas((msg + "Stages Activas:"), self._create_stager)
            return True

        else:
            log_simple(f"Error de configuración")
            return False

    def _validate_min_workers(self) -> bool:
        if not self._workers_order:
            log_simple("ERROR: No hay configuración de workers disponible")
            return False
        elif _min_workers.issubset(self._all_workers):
            return True
        else:
            workers_missing = _min_workers - self._all_workers
            log_simple(f"Faltan: {workers_missing} de los '{len(_min_workers)}' workers mínimos para el pipeline")
            return False
