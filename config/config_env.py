import os
import sys
from utils import paths, file_handler
from services import  system_service

current_dir = os.path.dirname(os.path.abspath(__file__))
project_base = os.path.dirname(current_dir)
# paths.set_project_root(project_base)
so = next(iter(system_service.SO.keys()))
print(f"project_root: '{project_base}', SISTEMA OPERATIVO: '{so}'")

libs_ver = ["cmake-", "ninja-", "python-3.12.1", "numpy-1.26.4", "conda-26.7.2", "opencv-4.11"]

python_path = sys.executable
print(f"PYTHON PATH: '{python_path}'")
norm_python_path = os.path.normpath(python_path)
intel_prefix_path = os.path.dirname(os.path.dirname(norm_python_path))
print(f"intel env: '{intel_prefix_path}'")
all_paths = paths.build_from_dir(objetives=libs_ver, search_root=intel_prefix_path, parent=["conda-meta"], kind_search="reference")

for lib, path in zip(libs_ver, all_paths):
    print(f"LIB: '{lib[:-1]}', path: '{path}'")

num_requeriments = len(libs_ver)
num_found = len(all_paths)

if num_found != num_requeriments:
    sys.exit("NO SE ENCONTRARON LAS LIBRERIAS MÍNIMAS PARA EL PIPELINE")

presets = paths.build_from_dir(objetives=["CMakePresets.json"], search_root=project_base)
print(f"prests path: '{presets}'")

digital_presets = file_handler.load_json(*presets, "r")

for key, config_data in digital_presets.items():
    if key == "configurePresets":
        config_data
        print(f"prests keys: '{config_data}'")