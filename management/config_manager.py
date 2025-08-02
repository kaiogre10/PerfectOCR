# PerfectOCR/management/config_manager.py
import yaml
import os
import logging
from typing import Dict, Any

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

    # --- CONFIGURACIONES BASE ---
    
    def get_system_config(self) -> Dict[str, Any]:
        """Obtiene configuración del sistema."""
        return self.config.get('system', {})
    
    def get_paths_config(self) -> Dict[str, Any]:
        """Obtiene todas las rutas del sistema."""
        return self.config.get('paths', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Obtiene configuración de logging."""
        return self.config.get('logging', {})
    
    def get_enabled_outputs(self) -> Dict[str, bool]:
        """Obtiene flags de salida habilitados."""
        return self.config.get('enabled_outputs', {})
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Obtiene configuración de procesamiento."""
        return self.config.get('processing', {})
    
    def get_cleanup_config(self) -> Dict[str, Any]:
        """Obtiene configuración de limpieza."""
        return self.config.get('cleanup', {})

    # --- CONFIGURACIONES DE MÓDULOS ---
    
    def get_image_loader_config(self) -> Dict[str, Any]:
        """Obtiene configuración específica del cargador de imágenes."""
        return self.config.get('modules', {}).get('image_loader', {})
    
    def get_polygonal_config(self) -> Dict[str, Any]:
        """Obtiene configuración específica del módulo poligonal."""
        return self.config.get('modules', {}).get('polygonal', {})
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Obtiene configuración específica del preprocesamiento."""
        return self.config.get('modules', {}).get('preprocessing', {})
    
    def get_ocr_config(self) -> Dict[str, Any]:
        """Obtiene configuración específica del OCR (sin rutas)."""
        return self.config.get('modules', {}).get('ocr', {})
    
    def get_vectorization_config(self) -> Dict[str, Any]:
        """Obtiene configuración específica de vectorización."""
        return self.config.get('modules', {}).get('vectorization', {})
    
    # --- CONFIGURACIONES ESPECIALES ---
    
    def get_ocr_config_with_paths(self) -> Dict[str, Any]:
        """Obtiene configuración de OCR con rutas de modelos (para PaddleOCR)."""
        ocr_config = self.get_ocr_config()
        paths_config = self.get_paths_config()
        
        # Combinar configuración de OCR con rutas de modelos
        if 'paddleocr' in ocr_config and 'models' in paths_config:
            ocr_config['paddleocr'].update(paths_config['models']['paddle'])
        
        return ocr_config
    
    def get_max_workers(self) -> int:
        """Obtiene número de workers desde configuración de procesamiento."""
        return self.get_processing_config().get('max_workers', 4)
    
    def get_max_workers_for_cpu(self) -> int:
        """Obtiene el número óptimo de workers basado en CPU."""
        batch_config = self.get_processing_config().get('batch_processing', {})
        cpu_count = os.cpu_count() or 4
        max_cores = batch_config.get('max_physical_cores', 4)
        add_extra = batch_config.get('add_extra_worker', True)
        workers = min(max_cores, cpu_count - 1)
        if add_extra and workers < cpu_count:
            workers += 1
        return max(1, workers)
    