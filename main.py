# main.py
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from services import log_service

DEFAULT_CONFIG_FILE = os.path.join(PROJECT_ROOT, "config", "master_config.yaml")

from services.config_service import ConfigService
config_service = ConfigService(DEFAULT_CONFIG_FILE)
if config_service.test_config:
    log_service.log_simple("TEST CONFIG ACTIVADO, FINALIZANDO")
    sys.exit()

system_config = config_service.system_config

from services import storage_service
storage_service.storage_config(PROJECT_ROOT, system_config) # type: ignore

import services.system_service as system_service
import services.output_service as output_service
from app.main_builder import MainBuilder

def main():
    os.environ.update(config_service.env_config)
    log_service.setup_logging(PROJECT_ROOT)

    system_service.set_system_config(PROJECT_ROOT, system_config) # type: ignore
    system_service.clear_output_folders()

    workflow_report = system_service.count_and_plan()
    if not workflow_report and not config_service.no_activate_modules:
        system_service.clear_output_folders()
        log_service.log_simple("NO HAY INPUT PATHS")
        sys.exit()

    output_service.set_output_config(PROJECT_ROOT, system_config) # type: ignore
    
    main_builder = MainBuilder(config_service, PROJECT_ROOT)
    main_builder.activate_main(workflow_report)

    system_service.cleanup_project_cache()

if __name__ == "__main__":
    main()

# CONFIGURACIÓN DINÁMICA PARA EL USUARIO FINAL
# import psutil  # Librería estándar recomendada para detectar hardware real

# def obtener_configuracion_entorno_dinamico():
#     # 1. Detectar hardware real del usuario final
#     es_intel = "intel" in psutil.cpu_names()[0].lower() if hasattr(psutil, 'cpu_names') else False
#     nucleos_fisicos = psutil.cpu_count(logical=False) or 2  # Respaldo mínimo seguro de 2
#     ram_total_gb = psutil.virtual_memory().total / (1024 ** 3)
    
#     # 2. Configuración base ULTRA-SEGURA (Para cualquier CPU/RAM)
#     config = {
#         'OPENBLAS_NUM_THREADS': '1',     # NumPy ligero siempre corre mejor en 1 hilo en producción
#         'NUMEXPR_NUM_THREADS': '1',      # Evita hilos ocultos en Pandas/filtros
#         'FLAGS_use_mkldnn': '1',         # Activa oneDNN para PaddleOCR si hay soporte
#         'MKL_DYNAMIC': 'TRUE'            # Que el backend decida según el tamaño de la matriz
#     }
    
#     # 3. Estrategia adaptativa para la CPU (PaddleOCR masivo)
#     # Dejamos un núcleo libre si el usuario tiene más de 2 núcleos, para que su PC no se trabe
#     hilos_calculados = max(1, nucleos_fisicos - 1) if nucleos_fisicos > 2 else nucleos_fisicos
    
#     config['OMP_NUM_THREADS'] = str(hilos_calculated)
#     config['MKL_NUM_THREADS'] = str(hilos_calculated)
    
#     # 4. Estrategia adaptativa para la Memoria RAM del usuario
#     if ram_total_gb < 7.5:
#         # Modo de bajo consumo de RAM: Forzamos a Intel MKL a liberar memoria de inmediato
#         config['MKL_DISABLE_FAST_MM'] = '1'  # Evita fugas y optimiza el uso en PCs lentas
#     else:
#         # El usuario tiene suficiente RAM (>= 8GB): Activamos la retención rápida
#         config['MKL_DISABLE_FAST_MM'] = '0'  # Aceleración por reutilización de buffers en RAM
        
#     # 5. Parches exclusivos solo si el usuario tiene procesador Intel
#     if es_intel:
#         config['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'  # Anclaje físico eficiente
#         config['KMP_BLOCKTIME'] = '0'   # Duerme los hilos al instante para no gastar batería de laptops
#     else:
#         # En CPUs que no son Intel (como AMD Ryzen), estas variables no aplican
#         config['KMP_BLOCKTIME'] = '0'   # GNU OpenMP a veces la lee como respaldo
        
#     return config

# # --- APLICACIÓN EN TU MAIN ---
# # Correr esto en las primeras 8 líneas antes de cualquier importación pesada
# os.environ.update(obtener_configuracion_entorno_dinamico())
