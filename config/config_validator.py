# services/config_validator.py
import os
from typing import Dict, Any, List, Set, Tuple, FrozenSet
from functools import cached_property
from services.log_service import log_active_areas, log_simple, basic_exc_logger
from decimal import Decimal
# from contextlib import contextmanager
from services.system_service import get_so
import cv2

kf_range = (0, 6)
sc_range = (1, 13)
ELEMENTAL_WORKER = "image_loader"
det = "geometry_detector"
ocr_workers: Set[str] = set(["polygon_extractor", "paddle_wrapper", det])
full_ocr: FrozenSet[str] = frozenset(ocr_workers.union(["data_finder"]))
vect_min = "lineal"     # Es parte de los min_workers pero por motivos de deploy lo mantendremos fuera
min_workers: FrozenSet[str] = frozenset(ocr_workers.union(set([ELEMENTAL_WORKER]))) # ["text_refiner", "lineal", "vectorizer", "cos_sim", "table_structurer", "math_max", "data_collector"]

class ConfigValidator:
    """Valida de los parametros de configuración"""
    def __init__(self, project_root: str, config: Dict[str, Any]):
        self.config = config
        self.project_root = project_root
    
        # with self._run_params() as tested:
        #     log_simple(f"TESTED: {tested}")
            
        if not self._validate_config():
            self.config = {}
            del self.config
        else:
            self.config = self.config

    @cached_property
    def elemental_params(self) -> bool:                         # Condición mínima necesaria para que pueda arrancar el sistema
        return ELEMENTAL_WORKER in self.create_stager[0][1]

    @cached_property
    def active_full_ocr(self):
        return ocr_workers.issubset(self.all_workers)

    @cached_property
    def system_params(self):
        return self.config.get("system_params", {})

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
    def update_model(self) -> bool:
        return bool(self.deploy_settings.get("update_model"))
    
    @cached_property
    def test_wf_model(self) -> bool:
        test_wf_model = bool(self.deploy_settings.get("test_wf_model"))
        self.update_model = False if test_wf_model else self.update_model
        return test_wf_model
    
    @cached_property
    def db_local(self) -> bool:
        return bool(self.deploy_settings.get("postgre_local"))
    
    @cached_property
    def no_activate_modules(self) -> bool: # Parametro automátizado que permite arrancar el sistema para testear parametros de alto nivel sin crear objetos pesados de manera innecesaria
        """(deploy_mode == True) and (elemental_params == False)"""
        return (self.deploy_mode == True) and (self.elemental_params == False)
    
    @cached_property
    def not_run_system(self) -> bool:
        """(self.deploy_mode == False) and (self.elemental_params == False)"""
        return (self.deploy_mode == False) and (self.elemental_params == False)
    
    @cached_property
    def enabled_outputs(self) -> Dict[str, Dict[str, bool]]:
        if not self.system_paths["output_paths"] or self.no_activate_modules:
            return {}
        else:
            return self.config.get("enabled_outputs", {})
        
    @property
    def user_requests(self):
        return self.config.get("user_requests", {})
    
    @cached_property
    def _input_paths(self):
        _input_paths = self.user_requests.get("input_paths", {})
        _input_paths["images_names"] = set(_input_paths["images_names"])
        _input_paths["skip_names"] = set(_input_paths["skip_names"])
        return _input_paths
    
    @cached_property
    def input_paths(self) -> Dict[str, List[str]]:
        input_paths = self._input_paths
        return {} if (not input_paths["input_dirs"] and not self.deploy_mode) else input_paths

    @cached_property
    def payload_request(self) -> Dict[str, Any]:
        return self.user_requests.get("payload_request", {})

    @cached_property
    def workers_order(self) -> Dict[str, List[str]]:
        return self.config.get("pipeline_secuence", {})
    
    @cached_property
    def system_paths(self) -> Dict[str, Any]:
        system_paths = self.system_params.get("system_paths", {})
        extension = get_so()
        output_paths = system_paths["output_paths"]
        temp_path = system_paths["temp_path"]
        
        libs_path = system_paths.get("libs_path", "")
        containers = system_paths.get("containers", "")
        buffer_handler = system_paths.get("buffer_handler", "")
        system_paths["output_paths"] = [os.path.join(self.project_root, folder) for folder in output_paths]
        system_paths["temp_path"] = os.path.join(self.project_root, *temp_path)
        container_path = os.path.join(self.project_root, libs_path, (containers + extension))
        buffer_path = os.path.join(self.project_root, libs_path, (buffer_handler + extension))
        if not os.path.isfile(container_path) or not os.path.isfile(buffer_path):
            self.handle_memory = False
            basic_exc_logger("NO EXISTEN LOS BINARIOS SE MODIFCA A FALSE EL MANEJO DE MEMORIA")
            
        system_paths["containers"] = container_path
        system_paths["buffer_handler"] = buffer_path
        
        return system_paths

    @cached_property
    def _logs_debug(self):
        _logs_debug = self.config.get("log_debug", {})
        def _max_min_vals(type_list: List[int], lenght: int, limits: Tuple[int, int]) -> List[int]:
            if -1 in type_list:
                range_logs = limits
            elif 1 < lenght:
                min_val = max_val = type_list[0]
                boundaries: List[int] = [0, 0]
                for val in type_list[1:]:
                    if val < min_val:
                        min_val = val
                        boundaries[0] = min_val
                    elif val > max_val:
                        max_val = val
                        boundaries[1] = max_val
                range_logs = boundaries
            else:
                if type_list[0] >= limits[1]:
                    range_logs = [limits[1] + (limits[1] + 1)]
                else:
                    range_logs = [type_list[0] + (type_list[0] + 1)]
                    
            return list(range(range_logs[0], range_logs[1]))
        
        kf_list_log = _logs_debug["kf_list_log"]
        semantic_types_log = _logs_debug["semantic_types_log"]
        
        if _logs_debug.get("all_logs"):
            sc_range_logs = list(range(kf_range[0], kf_range[1]))
            kf_range_logs = list(range(sc_range[0], sc_range[1]))
            
        else:
             kf_range_logs = [] if not _logs_debug["key_fields"] else _max_min_vals(kf_list_log, len(kf_list_log), kf_range)
             sc_range_logs = [] if not _logs_debug["seman_clas"] else  _max_min_vals(semantic_types_log, len(semantic_types_log), sc_range)
        
        _logs_debug["kf_list_log"] = kf_range_logs
        _logs_debug["semantic_types_log"] = sc_range_logs
        _logs_debug["handle_memory"] = self.handle_memory
        return _logs_debug
    
    @cached_property
    def logs_debug(self) -> Dict[str, Any]:
        logs_debug = self._logs_debug
        if logs_debug.get("all_logs"):
            for key, value in logs_debug.items():
                if isinstance(value, bool):
                    logs_debug[key] = True
                elif isinstance(value, list):
                    continue
        return logs_debug
        
    @cached_property
    def _models_config(self) -> Dict[str, Any]:
        _models_config: Dict[str, Dict[str, Any]] = self.config.get("models_config", {})
        paddle_config: Dict[str, Any] = _models_config.get("paddle_config", {})
        wf_config: Dict[str, Any] = _models_config.get("wf_config", {})
        models_paths: Dict[str, str] = _models_config.get("models_paths", {})
        
        _models_dir = models_paths.get("models_dir", "")
        models_dir = os.path.join(self.project_root, _models_dir)
        
        det_model = models_paths.get("det_model", "")
        rec_model = models_paths.get("rec_model", "")
        paddle_path = models_paths.get("paddle_path", "")
        lang = paddle_config.get("lang", "")
        
        _models_config["paddle_config"]["det_model_dir"] = os.path.join(models_dir, paddle_path, det_model, lang)
        _models_config["paddle_config"]["rec_model_dir"] = os.path.join(models_dir, paddle_path, rec_model, lang)
        _models_config["paddle_config"]["activate_rec"] = True
        _models_config["paddle_config"]["activate_det"] = True
        
        matrix_path = wf_config.get("matrix_path", "")
        kf_path = wf_config.get("kf_path", "")
        
        kf_idx_name = wf_config.get("kf_idx", "")
        pkl_path_name = wf_config.get("pkl_path", "")
        
        _wf_path = models_paths.get("word_finder_path", "")
        wf_path = os.path.join(models_dir, _wf_path)
        _models_config["wf_config"]["wf_path"] = wf_path
        
        _models_config["wf_config"]["kf_idx"] = os.path.join(wf_path, kf_idx_name)
        _models_config["wf_config"]["pkl_path"] = os.path.join(wf_path, pkl_path_name)
        _models_config["wf_config"]["train_data"] = os.path.join(self.project_root, "core", "assets", "data.npy")
        _models_config["wf_config"]["matrix_folder"] = os.path.join(wf_path, matrix_path)
        _models_config["wf_config"]["kf_folder"] = os.path.join(wf_path, kf_path)
        
        _models_config["wf_config"]["test_wf_model"] = self.test_wf_model
        del _models_config["models_paths"]
        return _models_config
    
    @cached_property
    def models_config(self) -> Dict[str, Any]:
        if self.no_activate_modules:
            return {
                "models_config": self._models_config,
                "activate_wf": True,
                "update_model": self.update_model
            }

        if not self.all_workers:
            return {}

        if det not in self.all_workers:
            # log_simple("Configuración: Sin geometry_detector, no se cargan modelos")
            return {}

        if full_ocr.issubset(self.all_workers):
            # log_simple("Configuración: OCR completo + Word Finder")
            return {
                "models_config": self._models_config,
                "activate_wf": True,
                "update_model": self.update_model
            }

        if self.active_full_ocr:
            # log_simple("Configuración: OCR completo sin Word Finder")
            return {
                "models_config": self._models_config,
                "activate_wf": False,
                "update_model": self.update_model
            }

        # log_simple("Configuración: Solo modelo de detección")
        self._models_config["paddle_config"]["activate_rec"] = False
        return {
            "models_config": self._models_config,
            "activate_wf": False,
            "update_model": self.update_model
        }

    @cached_property
    def modules_config(self) -> Dict[str, Any]:
        return {} if self.no_activate_modules else self.config.get("modules", {})
    
    @cached_property
    def _img_prep_config(self):
        image_workers = self.workers_order["image_preparation_stager"]
        if "polygon_extractor" in image_workers:
            if not det in image_workers:
                self.workers_order["image_preparation_stager"].remove("polygon_extractor")
                
        _img_prep_config = self.modules_config.get("image_preparation", {})
        geometry_detect = _img_prep_config.get("geometry_detect", {})
        ink_enh = _img_prep_config.get("ink_enhancement", {})
        
        morph_kernel = geometry_detect["morph_kernel"]
        _img_prep_config["geometry_detect"]["morph_kernel"] = cv2.getStructuringElement(cv2.MORPH_CROSS, (morph_kernel[0], morph_kernel[1]))
        _img_prep_config["angle_corrector"]["white"] = ink_enh["white"]
        return _img_prep_config
    
    @cached_property
    def img_prep_config(self) -> Tuple[Dict[str, Any], List[str]]:
        if self.no_activate_modules or not self.create_stager[0][1]:
            return ({}, [])
        else:
            config_module = self._img_prep_config
            enabled_outputs = self.enabled_outputs.get("image_load_outputs", {})
            config_module.update(enabled_outputs)
            return config_module, self.workers_order["image_preparation_stager"]
            
    @cached_property
    def preprocessing_config(self)-> Tuple[Dict[str, Any], List[str]]:
        if self.no_activate_modules or not self.create_stager[1][1] or not self.active_full_ocr:
            return ({}, [])
        else:
            config_module = self.modules_config.get("preprocessing", {})
            enabled_outputs = self.enabled_outputs.get("preprocessing_outputs", {})
            config_module.update(enabled_outputs)
            return config_module, self.workers_order["preprocessing_stager"]
            
    @cached_property
    def _ocr_config(self):
        _ocr_config = self.modules_config.get("ocr", {})
        text_refine = _ocr_config.get("text_refine", {})
        num_passes = text_refine.get("num_passes", {})
        _ocr_config["text_refine"]["create_refiners"] = bool(num_passes > 0)
        return _ocr_config
        
    @cached_property
    def ocr_config(self) -> Tuple[Dict[str, Any], List[str]]:
        if self.no_activate_modules or not self.create_stager[2][1] or not self.active_full_ocr:
            return ({}, [])
        else:
            config_module = self._ocr_config
            config_module.update(self.logs_debug)
            config_module.update(self.enabled_outputs.get("ocr_outputs"))
            return config_module, self.workers_order["ocr_stager"]
    
    @cached_property
    def _vectorization_config(self) -> Dict[str, Any]:
        _vectorization_config = self.modules_config.get("vectorization", {})
        all_cols_name = self.payload_request["payload_cols"]
        math_max = _vectorization_config.get("math_max", {})
        row_tol = math_max.get("row_tol", "")
        if not isinstance(row_tol, str):
            if not isinstance(row_tol, float):
                del self.modules_config["vectorization"]
                basic_exc_logger(f"ERROR TYPO DESCONOCIDO: {row_tol}")
                return {}
            
            basic_exc_logger(f"REVISAR CONFIGURACIÓN SE ENCONTRÓ: {type(row_tol)}, DEBERÍA SER STRING")
            row_tol = str(row_tol)
            
        _vectorization_config["math_max"]["row_tol"] = Decimal(row_tol)
        _vectorization_config["math_max"]["columns"] = all_cols_name[:4]
        _vectorization_config["math_max"]["dec_cols_name"] = frozenset(all_cols_name[:3])
        _vectorization_config["collector"]["cols_name"] = all_cols_name
        
        return _vectorization_config
    
    @cached_property
    def vectorization_config(self) -> Tuple[Dict[str, Any], List[str]]:
        vect_stage = self.create_stager[3][1]
        if self.no_activate_modules or not vect_stage or not vect_min in vect_stage or not self.active_full_ocr:
            return ({}, [])
        else:
            config_module = self._vectorization_config
            enabled_outputs = self.enabled_outputs.get("vectorization_outputs", {})
            config_module.update(enabled_outputs)
            return config_module, self.workers_order["vectorization_stager"]
        
    @cached_property
    def stagers_config(self) -> Dict[str, Tuple[Dict[str, Any], List[str]]]:
        """
        Se ejecuta UNA SOLA VEZ en el primer llamado de todo el pipeline.
        Construye la estructura y congela el mapa. Los subsecuentes accesos
        de los workers leerán directamente de la memoria RAM sin ejecutar código.
        """
        return {
            "image_preparation_stager": self.img_prep_config,
            "preprocessing_stager": self.preprocessing_config,
            "ocr_stager":  self.ocr_config,
            "vectorization_stager": self.vectorization_config
        }

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

        elif self.not_run_system:
            log_simple("ERROR CRÍTICO, NO HAY IMAGE LOADER PARA PRODUCCIÓN")
            return False
        
        elif not self.deploy_mode and not self.handle_memory:
            log_simple("ACTIVAR MEMORIA DINÁMICA")
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
        if not self.workers_order:
            log_simple("ERROR: No hay configuración de workers disponible")
            return False
        elif min_workers.issubset(self.all_workers):
            return True
        else:
            workers_missing = min_workers - self.all_workers
            log_simple(f"Faltan: {workers_missing} de los '{len(min_workers)}' workers mínimos para el pipeline")
            return False
        
    # @contextmanager
    def _run_params(self):
        tested = None
        try:
            tested = any([self.not_run_system, bool(self.all_workers), self.active_full_ocr, bool(self.workers_order), bool(self.stagers_config), bool(self.models_config), bool(self.logs_debug)])
            yield tested
        finally:
            if tested:
                log_simple(f"CONTEXT MANAGER FINALIZADO")
                return tested