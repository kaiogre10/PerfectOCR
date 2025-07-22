# PerfectOCR/managment/config_manager.py
import yaml
import os
import logging
from typing import Dict, Any, Optional, List
import numpy as np

logger = logging.getLogger(__name__)

class ConfigManager:
    """Cargador centralizado que lee YAML y proporciona configuraciones específicas para cada coordinador."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_yaml_config()
        
    def _load_yaml_config(self) -> Dict[str, Any]:
        """Carga el archivo YAML principal."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
                return config
        except Exception as e:
            logger.error(f"Error cargando configuración desde {self.config_path}: {e}")
            raise

    def get_output_config(self) -> Dict[str, Any]:
        return self.config.get('output_config', {})
    
    def get_quality_assessment_rules(self) -> Dict[str, Any]:
        """Obtiene reglas de evaluación de calidad directamente del YAML."""
        return self.config.get('image_preparation', {}).get('quality_assessment_rules', {})
    
    def get_workflow_config(self) -> Dict[str, Any]:
        """Obtiene configuración del flujo de trabajo."""
        return self.config.get('workflow', {})
    
    def get_ocr_config(self) -> Dict[str, Any]:
        """Obtiene configuración de motores OCR."""
        return self.config.get('ocr', {})
    
    def get_max_workers(self) -> int:
        """Obtiene workers desde configuración."""
        return self.get_workflow_config().get('max_workers', 4)
    
    def get_max_workers_for_cpu(self) -> int:
        """Obtiene el número óptimo de workers basado en CPU."""
        batch_config = self.config.get('batch_processing', {})
        cpu_count = os.cpu_count() or 4
        max_cores = batch_config.get('max_physical_cores', 4)
        add_extra = batch_config.get('add_extra_worker', True)  
        workers = min(max_cores, cpu_count - 1)
        if add_extra and workers < cpu_count:
            workers += 1
        return max(1, workers)
    
    # --- MÉTODOS ESPECÍFICOS PARA CADA COORDINADOR ---
    
    def get_polygonal_config(self) -> Dict[str, Any]:
        """Obtiene configuración de extracción de los polígonos"""
        output_config = self.config.get('output_config', {})
        return {
            'geometry': self.config.get('polygonal', {}),
            'output_flags': output_config.get('enabled_outputs', {})
        }
    
    def get_preprocessing_coordinator_config(self) -> Dict[str, Any]:
        """Proporciona la configuración completa para PreprocessingCoordinator."""
        return {
            'max_workers': self.get_max_workers(),
            'workflow': self.get_workflow_config(),
            'quality_assessment_rules': self.get_quality_assessment_rules(),
            'output_config': self.config.get('output_config', {})
        }
    
    def get_ocr_coordinator_config(self) -> Dict[str, Any]:
        """Proporciona la configuración completa para OCRCoordinator."""
        ocr_config = self.get_ocr_config()
        output_config = self.get_output_config('output_config', {})
        
        return {
            'ocr_config': ocr_config,
            'output_flags': output_config.get('enabled_outputs', {})
        }
    
    def get_tensor_coordinator_config(self) -> Dict[str, Any]:
        """Proporciona la configuración completa para TensorCoordinator."""
        output_config = self.config.get('output_config', {})
        return {
            'vectorization_process': self.config.get('vectorization_process', {}),
            'output_flags': output_config.get('enabled_outputs', {})
        }
            
    def get_text_cleaning_config(self) -> Dict[str, Any]:
        """Obtiene configuración de limpieza de texto."""
        output_config = self.config.get('output_config', {})
        
        return {
            'text_cleaning': self.config.get('text_cleaning', {}),
            'output_flags': output_config.get('enabled_outputs', {})
        }
    