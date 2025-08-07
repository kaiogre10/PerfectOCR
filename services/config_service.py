# PerfectOCR/management/config_manager.py
import yaml
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ConfigService:
    """Fragmentador centralizado que carga YAML una sola vez y proporciona configuraciones específicas para cada módulo."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_yaml_config()
        
    def load_yaml_config(self) -> Dict[str, Any]:
        """Carga el YAML una sola vez."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
                return config
        except Exception as e:
            logger.error(f"Error cargando configuración desde {self.config_path}: {e}")
            raise
    
    def system_config(self) -> Dict[str, Any]:
        """Obtiene configuración del sistema."""
        system_config = self.config.get('system', {})
        return system_config
        
    @property
    def logging_config(self) -> Dict[str, Any]:
        """Obtiene configuración de logging."""
        logging_config = self.config.get('logging', {})
        return logging_config
    
    @property
    def paths_config(self) -> Dict[str, Any]:
        """Obtiene todas las rutas del sistema."""
        paths_config = self.config.get('paths', {})
        return paths_config

    @property
    def enabled_outputs(self) -> Dict[str, Any]:
        """Obtiene flags de salida habilitados."""
        enabled_outputs = self.config.get('enabled_outputs', {})
        return enabled_outputs

    @property
    def processing_config(self) -> Dict[str, Any]:
        """Obtiene configuración de procesamiento."""
        processing_config = self.config.get('processing', {})
        return processing_config
    
    @property
    def modules_config(self) -> Dict[str, Any]:
        """Obtiene configuración de módulos."""
        modules_config = self.config.get('modules', {})
        return modules_config

    @property
    def cleanup_config(self) -> Dict[str, Any]:
        """Obtiene configuración de limpieza."""
        cleanup_config = self.config.get('cleanup', {})
        return cleanup_config
    
    @property
    def pipeline_config(self) -> Dict[str, Any]:
        """Obtienen el orden de ejecución de los workers"""
        pipeline_config = self.config.get('pipeline', {})
        return pipeline_config
    
    @property
    def workflow_config(self) -> Dict[str, Any]:
        """Obtiene configuración del workflow."""
        workflow_config = self.config.get('system', {})
        return workflow_config

    @property
    def paddle_config(self) -> Dict[str, Any]:
        """Obtiene la configuración global para Paddle"""
        paddle_config = self.config.get('paddle_config', {})
        return paddle_config

    @property
    def paddle_det_config(self) -> Dict[str, Any]:
        """Devuelve la configuración de PaddleOCR para detección (GeometryDetector)."""
        paddle_det = self.paddle_config
        det_model_dir = self.paddle_config.get('det_model_dir', "")
        if det_model_dir:
            paddle_det['det_model_dir'] = det_model_dir
        return paddle_det

    @property
    def paddle_rec_config(self) -> Dict[str, Any]:
        """Devuelve la configuración de PaddleOCR para PaddleWrapper."""
        paddle_rec = self.paddle_config
        rec_model_dir = self.paddle_config.get('rec_model_dir', "")
        if rec_model_dir:
            paddle_rec['rec_model_dir'] = rec_model_dir
        return paddle_rec


# FUNCIONES MÁS ESPECÍFICAS:
    
    @property
    def input_path(self) -> str:
        """Devuelve la ruta de la carpeta de entrada."""
        input_path = self.paths_config.get('input_folder', "")
        return input_path

    @property
    def output_path(self) -> str:
        """Devuelve la ruta de la carpeta de salida."""
        output_path = self.paths_config.get('output_folder', "")
        return output_path 

    @property
    def manager_config(self) -> Dict[str, Any]:
        """ Devuelve el paquete estándar de configuraciones de los managers"""
        output_path = self.paths_config.get('output_folder', "")
        output_flags = self.enabled_outputs.get('output_flag', {})
        pipeline = self.pipeline_config.get('secuence', [])
        
        manager_stage_config = {
            "output_folder": output_path, #str
            "output_flag": output_flags, #dict
            'secuence': pipeline #dict
        }
        return manager_stage_config

    @property
    def image_loader_config(self) -> Dict[str, Any]:
        """Devuelve la configuración del image_loader con la ruta de entrada."""
        image_loader_params = self.modules_config.get('image_loader', {})
        paddle_det_config = self.paddle_config.get('det_model_dir', "")
        image_loader_config = {
            "image_loader": image_loader_params,
            "paddle_det": paddle_det_config
        }
        return image_loader_config
        
    @property
    def preprocessing_config(self) -> Dict[str, Any]:
        """Obtiene configuración específica del preprocesamiento."""
        preprocessing_config = self.modules_config.get('preprocessing', {})
        return preprocessing_config 
    
    @property
    def vectorization_config(self) -> Dict[str, Any]:
        """Obtiene configuración específica de vectorización."""
        vectorization_config =  self.modules_config.get('vectorization', {})
        return vectorization_config
                                
    @property
    def max_workers(self) -> int:
        """Obtiene número de workers desde configuración de procesamiento."""
        return self.processing_config.get('max_workers', {})
    
    @property
    def max_workers_for_cpu(self) -> int:
        """Obtiene el número óptimo de workers basado en CPU."""
        batch_config = self.processing_config.get('batch_processing', {})
        cpu_count = os.cpu_count() or 4
        max_cores = batch_config.get('max_physical_cores', {})
        add_extra = batch_config.get('add_extra_worker', True)
        workers = min(max_cores, cpu_count - 1)
        if add_extra and workers < cpu_count:
            workers += 1
        return max(1, workers)
    
    @property
    def roots_config(self) -> Dict[str, Any]:
        """Obtiene configuración de rutas para limpieza."""
        cleanup_config = self.cleanup_config
        return {
            'folders_to_empty': cleanup_config.get('folders_to_empty', ""),
            'folder_names_to_delete': cleanup_config.get('folder_extensions_to_delete', []),
            'project_root': self.workflow_config.get('project_root', "")
        }