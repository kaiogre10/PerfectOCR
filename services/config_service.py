# services/config_service.py
import yaml
import os
import logging
from typing import Dict, Any, cast
from config.config_models import MasterConfig

logger = logging.getLogger(__name__)

class ConfigService:
    """Fragmentador centralizado con validación robusta y flexibilidad."""
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.validated_config = self._load_and_validate_yaml(config_path)
        self.config = self.validated_config.model_dump()
        
    def _load_and_validate_yaml(self, config_path: str) -> MasterConfig:
        """Carga YAML y valida con Pydantic - ROBUSTEZ."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                raw = yaml.safe_load(f) or {} 
            if not isinstance(raw, dict):
                raise TypeError(f"Config raíz debe ser dict, recibido: {type(raw).__name__}")
            typed_raw = cast(Dict[str, Any], raw)
            return MasterConfig.model_validate(typed_raw)
            
        except Exception as e:
            logger.error(f"Error validando configuración desde {self.config_path}: {e}")
            raise
    
    @property
    def manager_config(self) -> Dict[str, Any]:
        """Devuelve el paquete estándar de configuraciones de los managers"""
        return {
            "output_folder": self.output_path,
            "output_flag": self.enabled_outputs,
            'secuence': self.processing_config.get('pipeline_secuence', {})
        }
    
    @property
    def paths_config(self) -> Dict[str, Any]:
        """Obtiene todas las rutas del sistema."""
        return self.config.get('paths', {})
    
    @property
    def enabled_outputs(self) -> Dict[str, bool]:
        """Obtiene flags de salida habilitados."""
        return self.config.get('enabled_outputs', {})
    
    @property
    def processing_config(self) -> Dict[str, Any]:
        """Obtiene configuración de procesamiento."""
        return self.config.get('processing', {})
    
    @property
    def modules_config(self) -> Dict[str, Any]:
        """Obtiene configuración de módulos."""
        return {
            "modules": self.config.get("modules", {}),
            "enabled_outputs": self.enabled_outputs, 
        }
    
    @property
    def paddle_config(self) -> Dict[str, Any]:
        """Obtiene la configuración global para Paddle"""
        return self.config.get('paddle_config', {})
    
    @property
    def validated_paddle_config(self):
        """Acceso directo al objeto Pydantic validado."""
        return self.validated_config.paddle_config
        
    @property
    def validated_modules_config(self):
        """Acceso directo al objeto Pydantic validado."""
        return self.validated_config.modules
    
    @property
    def input_path(self) -> str:
        """Devuelve la ruta de la carpeta de entrada."""
        return self.paths_config.get('input_folder', "")
    
    @property
    def output_path(self) -> str:
        """Devuelve la ruta de la carpeta de salida."""
        return self.paths_config.get('output_folder', "")
        
    @property
    def image_loader_config(self) -> Dict[str, Any]:
        """Devuelve la configuración del image_loader con la ruta de entrada."""
        return {
            "image_loader": self.modules_config.get('image_loader', {}),
        }

    @property
    def preprocessing_config(self) -> Dict[str, Any]:
        """Obtiene configuración específica del preprocesamiento."""
        return self.modules_config.get('preprocessing', {})
    
    @property
    def ocr_config(self) -> Dict[str, Any]:
        """Obtiene configuración específica del OCR."""
        return self.modules_config.get('ocr', {})
    
    @property
    def vectorization_config(self) -> Dict[str, Any]:
        """Obtiene configuración específica de vectorización."""
        return self.modules_config.get('vectorization', {})
    
    @property
    def max_workers(self) -> int:
        """Obtiene número de workers desde configuración de procesamiento."""
        return self.processing_config.get('max_workers', 4)
    
    @property
    def max_workers_for_cpu(self) -> int:
        """Obtiene el número óptimo de workers basado en CPU."""
        batch_config = self.processing_config.get('batch_processing', {})
        cpu_count = os.cpu_count() or 4
        max_cores = batch_config.get('max_physical_cores', 4)
        add_extra = batch_config.get('add_extra_worker', True)
        workers = min(max_cores, cpu_count - 1)
        if add_extra and workers < cpu_count:
            workers += 1
        return max(1, workers)

    @property
    def paddle_det_config(self) -> Dict[str, Any]:
        """
        Devuelve la configuración fusionada para el modelo de detección de Paddle.
        Incluye solo los parámetros generales relevantes y la ruta del modelo de detección.
        """
        paddle_config = self.paddle_config
        det_model = paddle_config.get('models', {})
        det_model_path = det_model.get("det_model_dir", "")
        
        return {
            "det_model_dir": det_model_path,
            "use_angle_cls": paddle_config.get("use_angle_cls", False),
            "show_log": paddle_config.get("show_log", False),
            "use_gpu": paddle_config.get("use_gpu", False),
            "enable_mkldnn": paddle_config.get("enable_mkldnn", True),
            "lang": paddle_config.get("lang", "es"),
        }
            
        
    @property
    def paddleocr(self) -> Dict[str, Any]:
        """
        Devuelve la configuración fusionada para el modelo de reconocimiento de Paddle.
        Incluye solo los parámetros generales relevantes y la ruta del modelo de detección.
        """
        paddleocr = self.paddle_config
        rec_model = paddleocr.get('models', {})
        rec_model_path = rec_model.get("rec_model_dir", "")
        
        return {
            "rec_model_dir": rec_model_path,
            "use_angle_cls": paddleocr.get("use_angle_cls", False),
            "show_log": paddleocr.get("show_log", False),
            "use_gpu": paddleocr.get("use_gpu", False),
            "enable_mkldnn": paddleocr.get("enable_mkldnn", True),
            "lang": paddleocr.get("lang", "es"),
        }