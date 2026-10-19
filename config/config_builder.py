# config/config_builder.py
import os 
from typing import Dict, Any, List, Tuple, Optional
from functools import cached_property
from services.log_service import basic_exc_logger
from utils.image_utils import configure_kernel
from utils.paths import set_project_root, build_from_dir
from services import system_service
from core.assets.assets import FULL_OCR
from core.assets.patterns import PLACEHOLDER_PATTERN, extension_suffix
from domain.class_models import DataKeys, StageKeys, Params, Workers
from config.config_logger import ConfigLogger
from config.base_config import BaseConfig

_placeholder_pattern = PLACEHOLDER_PATTERN
_extension_suffix = extension_suffix
_full_ocr = FULL_OCR

class ConfigBuilder:
    """Construye de los parametros de configuración"""
    def __init__(self, project_root: str):
        self.project_root = project_root
        self._config: Optional[BaseConfig] = None
        if not self.config.validate_config():
            self.config = {}
            del self.config

        self.so = next(iter(system_service.SO.keys()))            
        set_project_root(self.project_root)

    @cached_property
    def config(self):
        if self._config is not None and not self._config:
            self.config = {}
            return None
        
        if self._config is None:
            self._config = BaseConfig(self.project_root)
        return self._config
            
    @property
    def _config_logging(self):
        log_config = self.config.log_config
        enabled_outputs = {} if not self._activate_outputs else self.config.enabled_outputs
        
        config_dict = {"log_config": log_config, "enabled_outputs": enabled_outputs}
        _config_logging = ConfigLogger(config_dict)
        return _config_logging

    @property
    def log_params(self):
        return self._config_logging.log_params

    @property
    def enabled_outputs(self):
        return self._config_logging.enabled_outputs

    @property
    def input_paths(self) -> Dict[str, List[str]]:
        return self.config.input_paths    
        
    @cached_property
    def logs_debug(self) -> Dict[str, Any]:
        logs_debug = self._config_logging.logs_debug
        logs_debug["handle_memory"] = self.config.handle_memory
        return logs_debug

    @cached_property
    def _system_paths(self) -> Dict[str, Any]:
        _system_paths = self.config.system_params.get("system_paths", {})
        output_paths = _system_paths["output_paths"]
        temp_path = _system_paths["temp_path"]
        
        _system_paths["output_paths"] = build_from_dir(output_paths)
        _system_paths["temp_path"] = os.path.join(self.project_root, *temp_path)

        if self.config.compile_cython:
            _comp_funcs_file = _system_paths["comp_funcs_file"]
            comp_funcs_name = ".".join(_comp_funcs_file)

            del_buils = os.path.join(self.project_root, "build")
            del_path = os.path.join(self.project_root, *_comp_funcs_file[:-1])
            _system_paths["del_files"] = [[self.so, ".c"], [del_path], [del_buils]]

            _comp_funcs_file = os.path.join(self.project_root, *_comp_funcs_file)
            comp_funcs_name = _extension_suffix.sub("", comp_funcs_name)

            _system_paths["comp_funcs_name"] = comp_funcs_name
            _system_paths["comp_funcs_file"] = _comp_funcs_file


        elif not self.config.handle_memory:
            return _system_paths

        else:
            buffer_path = os.path.join(self.project_root, "libraries", ("buffer_handler" + self.so))
            if not os.path.isfile(buffer_path):
                basic_exc_logger("NO EXISTEN LOS BINARIOS SE MODIFCA A FALSE EL MANEJO DE MEMORIA")
                self.config.handle_memory = False

            _system_paths["buffer_handler"] = buffer_path

        return _system_paths

    @property
    def system_paths(self):
        return self._system_paths

    @property
    def specific_deletes(self):
        """Limpia directorios o archivos especificos"""
        components_dir = os.path.join(self.project_root, "components", "compiled")

        dirs = build_from_dir(objetives=["__pycache__"], parent=[self.project_root], skip_names=['third_party', 'models', "include", "src", 'documentation'])
        files = build_from_dir(objetives=[".cpp", self.so], parent=[components_dir], with_extensions=True, skip_names=[".py", ".txt", ".h", ".hpp", ".c"])
        print(f"FILES: {files}, DIRS: {dirs}")
        return dirs, files

    @property
    def _activate_outputs(self) -> bool:
        return bool(self._system_paths["output_paths"] and not self.config.no_activate_modules)

    @cached_property
    def _models_config(self) -> Dict[str, Any]:
        _models_config: Dict[str, Dict[str, Any]] = self.config.models_config
        paddle_config: Dict[str, Any] = _models_config.get("paddle_config", {})
        wf_config: Dict[str, Any] = _models_config.get("wf_config", {})
        models_paths: Dict[str, str] = _models_config.get("models_paths", {})
        
        _models_dir = models_paths.get("models_dir", "")
        paddle_path = models_paths.get("paddle_path", "")
        lang = paddle_config.get("lang", "")

        paddle_paths = build_from_dir(objetives=[lang], parent=[_models_dir, paddle_path], skip_names=["cls"])

        paddle_config.update({
            "det_model_dir": paddle_paths[1],
            "rec_model_dir": paddle_paths[0],
            "activate_rec": True,
            "activate_det": True,
        })
        
        matrix_path = wf_config.get(Params.MATRIX_PATH, "")
        kf_path = wf_config.get(Params.KF_PATH, "")
        pkl_path_name = wf_config.get(Params.PKL_PATH, "")
        _wf_path = models_paths.get("word_finder_path", "")
        ngrams_len = wf_config[Params.CHAR_NGRAMS]

        wf_paths = build_from_dir(objetives=[pkl_path_name, matrix_path, kf_path], parent_include=True, parent=[_models_dir, _wf_path])

        wf_config.update({
            Params.WF_PATH: wf_paths[-1],
            Params.PKL_PATH: wf_paths[0],
            Params.MATRIX_FOLDER: wf_paths[1],
            Params.KF_FOLDER: wf_paths[2],
            "all_kf_idx": configure_kernel(ngrams_len[1], ngrams_len[0]),
            "test_wf_model": self.config.test_wf_model,
        })

        del _models_config["models_paths"]
        return _models_config
    
    @cached_property
    def models_config(self) -> Dict[str, Any]:
        if self.config.no_activate_modules:
            return {
                "models_config": self._models_config,
                "activate_wf": True,
                "update_model": self.config.update_model
            }

        if not self.config.all_workers:
            return {}

        if Workers.GEOMETRY_DETECTOR not in self.config.all_workers:
            # log_simple("Configuración: Sin geometry_detector, no se cargan modelos")
            return {}

        if _full_ocr.issubset(self.config.all_workers):
            # log_simple("Configuración: OCR completo + Word Finder")
            return {
                "models_config": self._models_config,
                "activate_wf": True,
                "update_model": self.config.update_model
            }

        if self.config.active_full_ocr:
            # log_simple("Configuración: OCR completo sin Word Finder")
            return {
                "models_config": self._models_config,
                "activate_wf": False,
                "update_model": self.config.update_model
            }

        # log_simple("Configuración: Solo modelo de detección")
        self._models_config["paddle_config"]["activate_rec"] = False
        return {
            "models_config": self._models_config,
            "activate_wf": False,
            "update_model": self.config.update_model
        }

    @cached_property
    def modules_config(self) -> Dict[str, Any]:
        _modules_config = self.config.modules_config
        _modules_config["time_worker_log"] = self.logs_debug.get("time_worker_log")
        return _modules_config

    @cached_property
    def _img_prep_config(self):
        _img_prep_config = self.modules_config.get("image_preparation", {})
        image_workers = self.config.workers_order[StageKeys.IMGPREP_KEY]
        if Workers.POLYGON_EXTRACTOR in image_workers:
            if not Workers.GEOMETRY_DETECTOR in image_workers:
                self.config.workers_order[StageKeys.IMGPREP_KEY].remove(Workers.POLYGON_EXTRACTOR)
                
        _img_prep_config = self.modules_config.get("image_preparation", {})
        geometry_detect = _img_prep_config.get("geometry_detect", {})
        
        morph_kernel = geometry_detect["morph_kernel"]
        _img_prep_config["geometry_detect"]["morph_kernel"] = configure_kernel(morph_kernel[0], morph_kernel[1])
        return _img_prep_config
    
    @cached_property
    def img_prep_config(self) -> Tuple[Dict[str, Any], List[str]]:
        if self.config.no_activate_modules or not self.config.create_stager[0][1]:
            return {}, []
        else:
            config_module = self._img_prep_config
            enabled_outputs = self.enabled_outputs.get("image_preparation_outputs", {})
            config_module.update(enabled_outputs)
            return config_module, self.config.workers_order[StageKeys.IMGPREP_KEY]
            
    @cached_property
    def preprocessing_config(self)-> Tuple[Dict[str, Any], List[str]]:
        if self.config.no_activate_modules or not self.config.create_stager[1][1] or not self.config.active_full_ocr:
            return {}, []
        else:
            config_module = self.modules_config.get("preprocessing", {})
            enabled_outputs = self.enabled_outputs.get("preprocessing_outputs", {})
            config_module.update(enabled_outputs)
            return config_module, self.config.workers_order[StageKeys.PREPRO_KEY]
            
    @cached_property
    def _ocr_config(self):
        _ocr_config = self.modules_config.get("ocr", {})
        text_refine = _ocr_config.get("text_refine", {})
        num_passes = text_refine.get("num_passes", 0)
        _ocr_config["text_refine"]["create_refiners"] = (num_passes > 0)
        return _ocr_config
        
    @cached_property
    def ocr_config(self) -> Tuple[Dict[str, Any], List[str]]:
        if self.config.no_activate_modules or not self.config.create_stager[2][1] or not self.config.active_full_ocr:
            return {}, []
        else:
            config_module = self._ocr_config
            config_module.update(self.logs_debug)
            config_module.update(self.enabled_outputs.get("ocr_outputs", {}))
            return config_module, self.config.workers_order[StageKeys.OCR_KEY]
    
    @cached_property
    def _vectorization_config(self) -> Dict[str, Any]:
        _vectorization_config: Dict[str, Any] = self.modules_config.get("vectorization", {})
        collector = _vectorization_config.get(Workers.COLLECTOR, {})
        placeholder = collector.get("placeholder")
        if not isinstance(placeholder, str) or not _placeholder_pattern.search(placeholder):
            basic_exc_logger(f"NO HAY SEPARADOR VÁLIDO")
            return {}
        
        sc_matrix_type = collector.get("sc_matrix_type")
        _vectorization_config[Workers.MATH_MAX] = {"dec_cols_name": frozenset([DataKeys.cantidad_art.value, DataKeys.precio_unitario.value, DataKeys.costo_tran.value]), "sc_matrix_type": sc_matrix_type}
        
        return _vectorization_config
    
    @cached_property
    def vectorization_config(self) -> Tuple[Dict[str, Any], List[str]]:
        vect_stage = self.config.create_stager[3][1]
        if self.config.no_activate_modules or not vect_stage or not Workers.LINEAL in vect_stage or not self.config.active_full_ocr:
            return {}, []
        else:
            config_module = self._vectorization_config
            enabled_outputs = self.enabled_outputs.get("vectorization_outputs", {})
            config_module.update(enabled_outputs)
            return config_module, self.config.workers_order[StageKeys.VECT_KEY]
        
    @cached_property
    def stagers_config(self) -> Dict[str, Tuple[Dict[str, Any], List[str]]]:
        """
        Se ejecuta UNA SOLA VEZ en el primer llamado de todo el pipeline.
        Construye la estructura y congela el mapa. Los subsecuentes accesos
        de los workers leerán directamente de la memoria RAM sin ejecutar código.
        """
        return {
            StageKeys.IMGPREP_KEY: self.img_prep_config,
            StageKeys.PREPRO_KEY: self.preprocessing_config,
            StageKeys.OCR_KEY: self.ocr_config,
            StageKeys.VECT_KEY: self.vectorization_config
        }
