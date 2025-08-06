# PerfectOCR/core/utils/batch_tools.py
from typing import Iterator, List, TypeVar
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)
T = TypeVar('T')

def chunked(iterable: List[T], chunk_size: int) -> Iterator[List[T]]:
    """Divide una lista en chunks de tamaño específico."""
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i:i + chunk_size]

def get_optimal_workers(num_images: int, max_physical_cores: int = 4) -> int:
    """Calcula el número óptimo de workers según el número de imágenes."""
    if num_images <= 1:
        return 1
    elif num_images <= 5:
        return min(2, max_physical_cores)
    else:
        # Para lotes grandes, usar todos los núcleos disponibles
        return min(max_physical_cores + 1, num_images)

def estimate_processing_time(num_images: int, avg_time_per_image: float = 45.0) -> Dict[str, Any]:
    """Estima tiempos de procesamiento para diferentes modos."""
    workers = get_optimal_workers(num_images)
    
    # Tiempo secuencial
    sequential_time = num_images * avg_time_per_image
    
    # Tiempo paralelo (con overhead)
    parallel_time = (num_images / workers) * avg_time_per_image * 1.15  # 15% overhead
    
    return {
        'sequential_minutes': sequential_time / 60,
        'parallel_minutes': parallel_time / 60,
        'workers': workers,
        'speedup': sequential_time / parallel_time
    }