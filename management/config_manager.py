# PerfectOCR/management/config_manager.py
import yaml
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConfigManager:
    """Fragmentador centralizado que carga YAML una sola vez y proporciona configuraciones específicas para cada módulo."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_yaml_config()
        
    def _load_yaml_config(self) -> Dict[str, Any]:
        """Carga el YAML una sola vez."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
                return config
        except Exception as e:
            logger.error(f"Error cargando configuración desde {self.config_path}: {e}")
            raise
    
    @property
    def _system_config(self) -> Dict[str, Any]:
        """Obtiene configuración del sistema."""
        system_config = self.config.get('system', {})
        return system_config
        
    @property
    def _logging_config(self) -> Dict[str, Any]:
        """Obtiene configuración de logging."""
        logging_config = self.config.get('logging', {})
        return logging_config
    
    @property
    def _paths_config(self) -> Dict[str, Any]:
        """Obtiene todas las rutas del sistema."""
        paths_config = self.config.get('paths', {})
        return paths_config

    @property
    def _enabled_outputs(self) -> Dict[str, Any]:
        """Obtiene flags de salida habilitados."""
        enabled_outputs = self.config.get('enabled_outputs', {})
        return enabled_outputs

    @property
    def _processing_config(self) -> Dict[str, Any]:
        """Obtiene configuración de procesamiento."""
        processing_config = self.config.get('processing', {})
        return processing_config
    
    @property
    def _modules_config(self) -> Dict[str, Any]:
        """Obtiene configuración de módulos."""
        modules_config = self.config.get('modules', {})
        return modules_config

    @property
    def _cleanup_config(self) -> Dict[str, Any]:
        """Obtiene configuración de limpieza."""
        cleanup_config = self.config.get('cleanup', {})
        return cleanup_config
    
# FUNCIONES MÁS ESPECÍFICAS:
    
    @property
    def _input_path(self) -> str:
        """Devuelve la ruta de la carpeta de entrada."""
        input_path = self._paths_config.get('input_folder', "")
        return input_path

    @property
    def _output_path(self) -> str:
        """Devuelve la ruta de la carpeta de salida."""
        output_path = self._paths_config.get('output_folder', "")
        return output_path 

    @property
    def _manager_output_config(self) -> Dict[str, Any]:
        """ Devuelve el paquete estándar de configuraciones de los managers"""
        output_path = self._paths_config.get('output_folder', "")
        output_flags = self._enabled_outputs.get('output_flag', {})
        
        manager_stage_config = {
            "output_folder": output_path,
            "output_flag": output_flags
        }
        return manager_stage_config

    @property
    def _image_loader_config(self) -> Dict[str, Any]:
        """Devuelve la configuración del image_loader con la ruta de entrada."""
        image_loader_params = self._modules_config.get('image_loader', {})
        input_path = self._paths_config.get('input_folder', "")
        paddle_config = self._paddle_detection_config
        image_loader_config = {
            "image_loader": image_loader_params,
            "input_folder": input_path,
            "paddle_det": paddle_config
        }
        return image_loader_config
        
    @property
    def _preprocessing_config(self) -> Dict[str, Any]:
        """Obtiene configuración específica del preprocesamiento."""
        preprocessing_config = self._modules_config.get('preprocessing', {})
        return preprocessing_config
    
    @property
    def _ocr_config(self) -> Dict[str, Any]:
        """Obtiene configuración específica del OCR (sin rutas)."""
        ocr_config = self._modules_config.get('ocr', {})
        return ocr_config 
    
    @property
    def _vectorization_config(self) -> Dict[str, Any]:
        """Obtiene configuración específica de vectorización."""
        vectorization_config =  self._modules_config.get('vectorization', {})
        return vectorization_config
        
    @property
    def _paddle_models(self) -> Dict[str, Any]:
        """Devuelve los modelos básivos de  PaddleOCR"""
        paddle_models = self._paths_config.get('paddlepaddle', {})
        return paddle_models 
    
    @property
    def _paddle_rec(self) -> Dict[str, Any]:
        """Devuelve la configuración de PaddleOCR para reconocimiento (PaddleWrapper)."""
        paddle_rec = self._ocr_config.get('paddleocr', {}).copy()
        rec_model_dir = self._paddle_models.get('rec_model_dir', "")
        if rec_model_dir:
            paddle_rec['rec_model_dir'] = rec_model_dir
        return paddle_rec
                        
    @property
    def _paddle_detection_config(self) -> Dict[str, Any]:
        """Devuelve la configuración de PaddleOCR para detección (GeometryDetector)."""
        paddle_det = self._ocr_config.get('paddleocr', {}).copy()
        det_model_dir = self._paddle_models.get('det_model_dir', "")
        if det_model_dir:
            paddle_det['det_model_dir'] = det_model_dir
        return paddle_det
    
    @property
    def max_workers(self) -> int:
        """Obtiene número de workers desde configuración de procesamiento."""
        return self._processing_config.get('max_workers', {})
    
    @property
    def _max_workers_for_cpu(self) -> int:
        """Obtiene el número óptimo de workers basado en CPU."""
        batch_config = self._processing_config.get('batch_processing', {})
        cpu_count = os.cpu_count() or 4
        max_cores = batch_config.get('max_physical_cores', {})
        add_extra = batch_config.get('add_extra_worker', True)
        workers = min(max_cores, cpu_count - 1)
        if add_extra and workers < cpu_count:
            workers += 1
        return max(1, workers)