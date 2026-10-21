from functools import cached_property
import os
from typing import Dict, Any, List, Set, Tuple, FrozenSet
from core.assets.assets import ELEMENTAL_WORKER, OCR_WORKERS, MIN_WORKERS
from services.log_service import log_simple
from config.config_loader import load_config
from config.config_logger import ConfigLogger
from services import system_service
from domain.class_models import Workers

_elemental_workers = ELEMENTAL_WORKER
_ocr_workers = OCR_WORKERS     # Es parte de los min_workers pero por motivos de deploy lo mantendremos fuera
_min_workers = MIN_WORKERS

class BaseConfig:
    def __init__(self, project_root: str):
        self.config = load_config(project_root)

        if not self.validate_config():
            self.config = {}
            del self.config
            
        self.env_config

    @cached_property
    def elemental_params(self) -> bool:
        return _elemental_workers in self.create_stager[0][1]

    @cached_property
    def active_full_ocr(self):
        return _ocr_workers.issubset(self.all_workers)

    @property
    def activate_det(self):
        return Workers.GEOMETRY_DETECTOR in self.all_workers

    @property
    def system_params(self):
        return self.config.get("system_params", {})

    @property
    def deploy_settings(self) -> Dict[str, bool]:
        return self.config.get("deploy_settings", {})

    @property
    def clean_project(self) -> bool:
        return bool(self.deploy_settings.get("clean_mode"))

    @cached_property
    def handle_memory(self) -> bool:
        return bool(self.deploy_settings.get("handle_memory"))

    @property
    def deploy_mode(self) -> bool:
        return bool(self.deploy_settings.get("deploy_mode"))

    @property
    def test_config(self) -> bool:
        return bool(self.deploy_settings.get("test_config"))

    @cached_property
    def update_model(self) -> bool:
        return bool(self.deploy_settings.get("update_model"))

    @cached_property
    def test_wf_model(self) -> bool:
        test_wf_model = bool(self.deploy_settings.get("test_wf_model"))
        self.update_model = False if test_wf_model else self.update_model
        return test_wf_model

    @property
    def db_local(self) -> bool:
        return bool(self.deploy_settings.get("postgre_local"))

    @property
    def compile_cython(self) -> bool:
        return bool(self.deploy_settings.get("compile_cython"))

    @cached_property
    def no_activate_modules(self) -> bool: # Parametro automátizado que permite arrancar el sistema para testear parametros de alto nivel sin crear objetos pesados de manera innecesaria
        """(deploy_mode == True) and (elemental_params == False)"""
        return (self.deploy_mode == True) and (self.elemental_params == False)

    @cached_property
    def not_run_system(self) -> bool:
        """(self.deploy_mode == False) and (self.elemental_params == False)"""
        return (self.deploy_mode == False) and (self.elemental_params == False)

    @cached_property
    def _env_config(self) -> Dict[str, str]:
        _env_config = self.config.get("env_config", {})
        oneapi_root = os.environ.get('ONEAPI_ROOT', "")

        _env_config['TBBROOT'] = os.path.join("tbb", "latest", "bin")
        _env_config['IPPROOT'] = os.path.join("ipp", "latest", "bin")

        if self.compile_cython:
            compiler_path = os.path.join("compiler", "latest", "bin")
            if os.path.isdir(oneapi_root) and os.path.isdir(compiler_path):
                if ".dll" in system_service.SO.keys():
                    compiler_intel_win = os.path.join(compiler_path, "icx-cl")
                    _env_config['CC'] = compiler_intel_win
                    _env_config['CXX'] = compiler_intel_win
                    _env_config['LDSHARED'] = f'{compiler_intel_win} /LD'
                else:
                    _env_config['CC'] = os.path.join(compiler_path, "icx")
                    _env_config['CXX'] = os.path.join(compiler_path, "icpx")

                _env_config['DISTUTILS_USE_SDK'] = '1'

        return _env_config

    @property
    def env_config(self) -> Dict[str, str]:
        return self._env_config

    @property
    def user_requests(self) -> Dict[str, Dict[str, Any]]:
        return self.config.get("user_requests", {})

    @cached_property
    def workers_order(self) -> Dict[str, List[str]]:
        return self.config.get("pipeline_secuence", {})

    @property
    def log_config(self) -> Dict[str, Any]:
        return self.config.get("log_config", {})

    @property
    def _enable_outputs(self) -> bool:
        _system_paths = self.system_params.get("system_paths", {})
        _output_paths =  _system_paths["output_paths"]
        return bool(_output_paths and not self.no_activate_modules)

    @cached_property
    def enable_outputs(self) -> Dict[str, bool]:
        return {} if not self._enable_outputs else self.config.get("enabled_outputs", {})

    @cached_property
    def input_paths(self) -> Dict[str, Any]:
        _input_paths = self.user_requests.get("input_paths", {})
        if not _input_paths["input_dirs"] and not self.deploy_mode:
            return {}
        _input_paths["images_names"] = set(_input_paths["images_names"])
        _input_paths["skip_names"] = set(_input_paths["skip_names"])
        return _input_paths

    @cached_property
    def models_config(self) -> Dict[str, Any]:
        return self.config.get("models_config", {})

    @cached_property
    def modules_config(self) -> Dict[str, Any]:
        return {} if self.no_activate_modules else self.config.get("modules", {})

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

    def validate_config(self) -> bool:
        mode = f'{"DEPLOY" if self.deploy_mode else "PRODUCTION"}' if not self.no_activate_modules else ("CONFIG TESTING" if self.test_config else "NO MODULES")
        msg: str = f"Modo: '{mode}', Validación: '{"COMPLETA" if not self.deploy_mode else "MÍNIMA"}' de la configuración."

        if not self.input_paths:
            log_simple("No hay rutas input")
            return False

        elif self.not_run_system:
            log_simple("ERROR CRÍTICO, NO HAY IMAGE LOADER PARA PRODUCCIÓN")
            return False

        elif self.no_activate_modules:
            ConfigLogger.log_active_areas(msg + " 'SOLO MODULOS DE ALTO NIVEL'")
            return True

        elif not self.deploy_mode and not self.handle_memory:
            log_simple("ACTIVAR MEMORIA DINÁMICA")
            return False

        elif self.deploy_mode:
            ConfigLogger.log_active_areas((msg + " Modulos:"), self.create_stager)
            return True

        elif not self.no_activate_modules and self.validate_min_workers():
            ConfigLogger.log_active_areas((msg + "Stages Activas:"), self.create_stager)
            return True

        else:
            log_simple(f"Error de configuración")
            return False

    def validate_min_workers(self) -> bool:
        if not self.workers_order:
            log_simple("ERROR: No hay configuración de workers disponible")
            return False
        elif _min_workers.issubset(self.all_workers):
            return True
        else:
            workers_missing = _min_workers - self.all_workers
            log_simple(f"Faltan: {workers_missing} de los '{len(_min_workers)}' workers mínimos para el pipeline")
            return False
