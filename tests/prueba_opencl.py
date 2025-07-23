import cv2
import logging
import numpy as np
import typing 
import os
import psutil
import threading
from multiprocessing import cpu_count

# Intentar importar MKL
try:
    import mkl
    HAS_MKL = True
except ImportError:
    HAS_MKL = False

logger = logging.getLogger(__name__)

def check_thread_info():
    """Muestra información completa sobre hilos disponibles"""
    
    print("=== INFORMACIÓN DE HILOS Y PROCESAMIENTO ===")
    
    # 1. Hilos de CPU físicos y lógicos
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    print(f"CPU cores físicos: {physical_cores}")
    print(f"CPU cores lógicos (con hyperthreading): {logical_cores}")
    print(f"multiprocessing.cpu_count(): {cpu_count()}")
    
    # 2. Variables de entorno que afectan hilos
    print("\n=== VARIABLES DE ENTORNO ===")
    thread_vars = {
        'OMP_NUM_THREADS': 'OpenMP threads',
        'MKL_NUM_THREADS': 'Intel MKL threads', 
        'OPENBLAS_NUM_THREADS': 'OpenBLAS threads',
        'NUMEXPR_NUM_THREADS': 'NumExpr threads',
    }
    
    for var, description in thread_vars.items():
        value = os.environ.get(var, 'No establecido')
        print(f"{var} ({description}): {value}")
    
    # 3. MKL threads si está disponible
    print(f"\n=== INTEL MKL ===")
    if HAS_MKL:
        try:
            # Obtener número actual de hilos MKL
            current_mkl_threads = mkl.get_max_threads()
            print(f"MKL está disponible")
            print(f"MKL threads actuales: {current_mkl_threads}")
            
            # Información adicional de MKL
            print(f"MKL domain threads: {mkl.domain_get_max_threads()}")
            
        except Exception as e:
            print(f"Error obteniendo info de MKL: {e}")
    else:
        print("MKL no está disponible en este entorno")
    
    # 4. Threading info de Python
    print(f"\n=== PYTHON THREADING ===")
    print(f"Active threads: {threading.active_count()}")
    print(f"Current thread: {threading.current_thread().name}")
    
    return {
        'physical_cores': physical_cores,
        'logical_cores': logical_cores,
        'mkl_available': HAS_MKL,
        'mkl_threads': mkl.get_max_threads() if HAS_MKL else None
    }

def check_opencl_info():
    """Información detallada de OpenCL"""
    
    print("\n=== INFORMACIÓN DE OpenCL ===")
    
    # Verificar soporte OpenCL
    has_opencl = cv2.ocl.haveOpenCL()
    print(f"OpenCL disponible en OpenCV: {has_opencl}")
    
    if has_opencl:
        # Estado actual
        is_enabled = cv2.ocl.useOpenCL()
        print(f"OpenCL habilitado: {is_enabled}")
        
        # Habilitar OpenCL si no está habilitado
        if not is_enabled:
            cv2.ocl.setUseOpenCL(True)
            print("OpenCL habilitado manualmente")
        
        # Test básico de funcionalidad
        try:
            # Crear una imagen de prueba para verificar OpenCL
            test_img = cv2.UMat(np.ones((100, 100), dtype=np.uint8))
            result = cv2.GaussianBlur(test_img, (5, 5), 0)
            if result is not None:
                print("Test básico de OpenCL: EXITOSO")
            else:
                print("Test básico de OpenCL: FALLÓ")
                
        except Exception as e:
            print(f"Error en test de OpenCL: {e}")
        
        # Información adicional disponible
        print(f"Versión OpenCV: {cv2.__version__}")
        
        # Verificar métodos disponibles en ocl
        ocl_methods = [method for method in dir(cv2.ocl) if not method.startswith('_')]
        print(f"Métodos OpenCL disponibles: {len(ocl_methods)}")
        
    else:
        print("OpenCL no está disponible")
        print("Posibles causas:")
        print("   - No hay GPU compatible")
        print("   - Drivers de GPU desactualizados")
        print("   - OpenCV compilado sin soporte OpenCL")
    
    return has_opencl

def test_performance():
    """Test básico de rendimiento"""
    print(f"\n=== TEST DE RENDIMIENTO ===")
    
    # Crear matriz de prueba
    size = 1000
    img = np.random.randint(0, 255, (size, size), dtype=np.uint8)
    
    import time
    
    # Test CPU
    start = time.time()
    result_cpu = cv2.GaussianBlur(img, (15, 15), 0)
    time_cpu = time.time() - start
    print(f"Tiempo CPU: {time_cpu:.4f} segundos")
    
    # Test GPU (si está disponible)
    if cv2.ocl.haveOpenCL():
        try:
            cv2.ocl.setUseOpenCL(True)
            
            # Convertir a UMat (GPU)
            img_gpu = cv2.UMat(img)
            
            start = time.time()
            result_gpu = cv2.GaussianBlur(img_gpu, (15, 15), 0)
            # Forzar sincronización
            result_gpu_cpu = result_gpu.get()
            time_gpu = time.time() - start
            
            print(f"🔹 Tiempo GPU (OpenCL): {time_gpu:.4f} segundos")
            
            if time_gpu > 0:
                speedup = time_cpu / time_gpu
                print(f"Speedup: {speedup:.2f}x")
                
                if speedup > 1:
                    print("GPU es más rápido que CPU")
                else:
                    print("CPU es más rápido (normal para operaciones pequeñas)")
            
        except Exception as e:
            print(f"Error en test de GPU: {e}")
    else:
        print("OpenCL no disponible para test de GPU")

# Ejecutar todas las verificaciones
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    thread_info = check_thread_info()
    opencl_available = check_opencl_info()
    test_performance()
    
    print(f"\n=== RESUMEN FINAL ===")
    print(f"Cores físicos: {thread_info['physical_cores']}")
    print(f"Cores lógicos: {thread_info['logical_cores']}")
    print(f"MKL disponible: {thread_info['mkl_available']}")
    print(f"OpenCL disponible: {opencl_available}")
