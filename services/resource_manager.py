# services/performance_service.py (más simple)
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class PerformanceService:
    """Servicio simple para gestión de rendimiento"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._setup_environment()
    
    def _setup_environment(self):
        """Configura variables de entorno básicas"""
        os.environ.update({
            'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '2', 
            'FLAGS_use_mkldnn': '1'
        })
    
    def get_optimal_workers(self, num_images: int) -> int:
        """Versión simplificada del cálculo de workers"""
        processing_config = self.config.get('processing', {})
        batch_config = processing_config.get('batch_processing', {})
        
        small_batch_limit = batch_config.get('small_batch_limit', 5)
        max_physical_cores = batch_config.get('max_physical_cores', 4)
        
        if num_images <= small_batch_limit:
            return 1
        else:
            return min(max_physical_cores, num_images)
    
    def estimate_processing_time(self, num_images: int) -> Dict[str, Any]:
        """Estimación simple de tiempo"""
        workers = self.get_optimal_workers(num_images)
        avg_time = 45.0  # segundos por imagen
        
        sequential_time = num_images * avg_time
        parallel_time = (num_images / workers) * avg_time * 1.15
        
        return {
            'sequential_minutes': sequential_time / 60,
            'parallel_minutes': parallel_time / 60,
            'workers': workers,
            'speedup': sequential_time / parallel_time
        }