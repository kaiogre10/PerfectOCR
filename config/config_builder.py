# config/config_builder.py
import os 
from typing import Dict, Any, List, Tuple, Optional
from functools import cached_property
from services.log_service import basic_exc_logger, log_simple
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
            self._config = {}
            del self._config
                    
        self.so = next(iter(system_service.SO.keys()))
        set_project_root(self.project_root)

    @property
    def config(self):
        if self._config is not None and not self._config:
            self._config = None
        
        if self._config is None:
            self._config = BaseConfig(self.project_root)

        return self._config
            
    @property
    def _config_logging(self):        
        config_dict = {"log_config": self.config.log_config, "enabled_outputs": self.config.enable_outputs}
        _config_logging = ConfigLogger(config_dict)
        return _config_logging

    @property
    def log_params(self):
        return self._config_logging.log_params

    @property
    def enabled_outputs(self):
        return self._config_logging.enabled_outputs

    @property
    def _no_del_output(self):
        return self._config_logging.no_del_output

    @property
    def input_paths(self) -> Dict[str, List[str]]:
        return self.config.input_paths
        
    @property
    def logs_debug(self) -> Dict[str, Any]:
        return self._config_logging.logs_debug

    @cached_property
    def _system_paths(self) -> Dict[str, Any]:
        _system_paths = self.config.system_params.get("system_paths", {})
        temp_path = _system_paths["temp_path"]
        _system_paths["no_del_output"] = self._no_del_output
        _output_paths_config = _system_paths["output_paths"]
        output_list = build_from_dir(objetives=_output_paths_config, search_root=self.project_root, kind_search="no_extensions")
        if not all(os.path.isdir(paht) for paht in output_list):
            log_simple("CRUZA DE NOMBRES EN DIRECTORIOS DE SALIDA DEBUG")

        new_output_paths: List[str] = []
        if not output_list and _output_paths_config:
            _output_paths_config.clear()
            for path in output_list:
                output_path = os.path.join(self.project_root, path)
                os.mkdir(output_path)
                print(f"DIRECTORIO OUTPUT CREADO: {output_path}")
                new_output_paths.append(output_path)

        _system_paths["output_paths"] = new_output_paths
        _system_paths["temp_path"] = os.path.join(self.project_root, *temp_path)

        if self.config.compile_cython:
            _comp_funcs_file = _system_paths["comp_funcs_file"]
            comp_funcs_name = ".".join(_comp_funcs_file)

            del_build = os.path.join(self.project_root, "build")
            del_path = os.path.join(self.project_root, *_comp_funcs_file[:-1])
            _system_paths["del_files"] = [[self.so, ".pyd", ".c"], [del_path], [del_build]]

            _comp_funcs_file = os.path.join(self.project_root, *_comp_funcs_file)
            comp_funcs_name = _extension_suffix.sub("", comp_funcs_name)

            _system_paths["comp_funcs_name"] = comp_funcs_name
            _system_paths["comp_funcs_file"] = _comp_funcs_file

        return _system_paths

    @property
    def system_paths(self):
        return self._system_paths

    @property
    def specific_deletes(self):
        """Limpia directorios o archivos especificos"""
        bidings_dir = os.path.join(self.project_root, "bidings")
        
        preserve_ext = [".py", ".pyi", ".txt", ".h", ".hpp", ".pyx", ".pxd", ".txt", ".cmake", ".json"]
        cython_thrash = [".cpp", ".pyd", self.so, ".c"]

        if not os.path.isdir(bidings_dir):
            raise FileNotFoundError(f"NO ES UNA RUTA VALIDA: '{bidings_dir}'")

        cython_plus_bins = build_from_dir(objetives=cython_thrash, parent=[bidings_dir], kind_search="extensions", skip_names=preserve_ext)
        if not cython_plus_bins:
            pass

        files = cython_plus_bins

        if self.config.compile_cython:
            comp_func_dir = os.path.join(self.project_root, "utils", "compiled_funcs")
            if not os.path.isdir(comp_func_dir):
                raise FileNotFoundError(f"NO ES UNA RUTA VALIDA: '{comp_func_dir}'")

            cython_bins = build_from_dir(objetives=cython_thrash, parent=[comp_func_dir], kind_search="extensions", skip_names=preserve_ext)
            if not cython_bins:
                pass

            files = files + cython_plus_bins

        dirs = build_from_dir(objetives=["__pycache__"], parent=[self.project_root], kind_search="no_extensions", skip_names=["third_party", "cmake", "models", "include", "src", "install", "input", "input2", "input3", "documentation", "libs"])
        build_path = os.path.join(self.project_root, "build")
        if os.path.isdir(build_path):
            dirs = dirs + [build_path]

        libraries_path = os.path.join(self.project_root, "libraries")
        if os.path.isdir(libraries_path):
            dirs = dirs + [libraries_path]

        return dirs, files

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

        if not self.config.activate_det:
            basic_exc_logger("Configuración: Sin geometry_detector, no se cargan modelos")
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

    @property
    def _activate_img_prep(self):
        if self.config.no_activate_modules or not self.config.create_stager[0][1]:
            return False
        return True

    @cached_property
    def _img_prep_config(self) -> Tuple[Dict[str, Any], List[str]]:
        _img_prep_config = self.modules_config.get("image_preparation", {})
        if not self._activate_img_prep:
            del _img_prep_config
            return {}, []
        
        image_workers = self.config.workers_order[StageKeys.IMGPREP_KEY]
        if self.config.activate_det:
            geometry_detect = _img_prep_config.get("geometry_detect", {})        
            morph_kernel = geometry_detect["morph_kernel"]
            geometry_detect["morph_kernel"] = configure_kernel(morph_kernel[0], morph_kernel[1])
            geometry_detect["bboxes_type"] = self.config.pipeline_dtypes.get("int32_type", b"")
            geometry_detect["centroids_type"] = self.config.pipeline_dtypes.get("float32_type", b"")
                        
        enabled_outputs = self.enabled_outputs.get("image_preparation_outputs", {})
        _img_prep_config.update(enabled_outputs)
        return _img_prep_config, image_workers
                
    @cached_property
    def preprocessing_config(self)-> Tuple[Dict[str, Any], List[str]]:
        if self.config.no_activate_modules or not self.config.create_stager[1][1] or not self.config.active_full_ocr:
            return {}, []
        else:
            config_module = self.modules_config.get("preprocessing", {})
            enabled_outputs = self.enabled_outputs.get("preprocessing_outputs", {})
            config_module.update(enabled_outputs)
            return config_module, self.config.workers_order[StageKeys.PREPRO_KEY]    
    
    @property
    def _activate_ocr(self):
        if self.config.no_activate_modules or not self.config.create_stager[2][1] or not self.config.active_full_ocr:
            return False
        return True
    
    @cached_property
    def _ocr_config(self) -> Tuple[Dict[str, Any], List[str]]:
        _ocr_config = self.modules_config.get("ocr", {})
        if not self._activate_ocr:
            del _ocr_config
            return {}, []
        
        text_refine = _ocr_config.get("text_refine", {})
        num_passes = text_refine.get("num_passes", 0)
        _ocr_config["text_refine"]["create_refiners"] = (num_passes > 0)

        _ocr_config.update(self.logs_debug)
        _ocr_config.update(self.enabled_outputs.get("ocr_outputs", {}))
        return _ocr_config, self.config.workers_order[StageKeys.OCR_KEY]
            
    @property
    def _activate_vect(self):
        vect_stage = self.config.create_stager[3][1]
        if self.config.no_activate_modules or not vect_stage or not Workers.LINEAL in vect_stage or not self.config.active_full_ocr:
            return False
        return True
        
    @cached_property
    def _vectorization_config(self) -> Tuple[Dict[str, Any], List[str]]:
        _vectorization_config: Dict[str, Any] = self.modules_config.get("vectorization", {})
        if not self._activate_vect:
            del _vectorization_config
            return {}, []
        
        collector = _vectorization_config.get(Workers.COLLECTOR, {})
        placeholder = collector.get("placeholder")
        if not isinstance(placeholder, str) or not _placeholder_pattern.search(placeholder):
            basic_exc_logger(f"NO HAY SEPARADOR VALIDO")
            return {}, []
        
        sc_matrix_type = self.config.pipeline_dtypes.get("image_type", b"")
        _vectorization_config[Workers.MATH_MAX] = {"dec_cols_name": frozenset([DataKeys.cantidad_art.value, DataKeys.precio_unitario.value, DataKeys.costo_tran.value]), "sc_matrix_type": sc_matrix_type}
        enabled_outputs = self.enabled_outputs.get("vectorization_outputs", {})
        _vectorization_config.update(enabled_outputs)
        _vect_stage = self.config.workers_order[StageKeys.VECT_KEY]    
        
        return _vectorization_config, _vect_stage
        
    @cached_property
    def stagers_config(self) -> Dict[str, Tuple[Dict[str, Any], List[str]]]:
        """
        Se ejecuta UNA SOLA VEZ en el primer llamado de todo el pipeline.
        Construye la estructura y congela el mapa. Los subsecuentes accesos
        de los workers leerán directamente de la memoria RAM sin ejecutar código.
        """
        return {
            StageKeys.IMGPREP_KEY: self._img_prep_config,
            StageKeys.PREPRO_KEY: self.preprocessing_config,
            StageKeys.OCR_KEY: self._ocr_config,
            StageKeys.VECT_KEY: self._vectorization_config
        }
