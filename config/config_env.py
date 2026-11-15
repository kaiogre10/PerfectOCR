import os
import sys
import time
from services import log_service
from utils import paths, file_handler
from typing import Dict
from core.assets.env_assets import libs_ver, requeriments_list, version_call, python_base

time0 = time.perf_counter()

project_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logger = log_service.log_temp_process(project_base)

conda_paths = paths.build_from_dir(objetives=["CondaPaths.json"], search_root=project_base)
if not conda_paths:
    logger.warning("NO HAY CONDA PATHS")
    sys.exit()

digital_presets = file_handler.load_json(*conda_paths, mode="r")
dict_items = digital_presets["configurePresets"][0]
cacheVariables: Dict[str, str] = dict_items.get("cacheVariables", {})
logger.debug(f"digital: {digital_presets}")

python_path = sys.executable

envs_path = os.path.dirname(os.path.normpath(python_path))
if os.path.exists(os.path.join(envs_path, "envs")):
    logger.error("NO SE ACTIVO EL INTERPRETE DEL CONDA ENV, REVISAR CONDA")
    sys.exit()

CONDA_PREFIX = os.path.dirname(envs_path)

cacheVariables.update({
    "CONDA_ENV": os.path.basename(CONDA_PREFIX),
    "CONDA_PREFIX": CONDA_PREFIX,
    "Python3_ROOT_DIR": CONDA_PREFIX,
    "CMAKE_PREFIX_PATH": CONDA_PREFIX,
})

validation_root = os.path.join(CONDA_PREFIX, "conda-meta")

_libs_ver = libs_ver

try:
    some_validated = paths.build_from_dir(objetives=_libs_ver, search_root=validation_root, kind_search="reference")
    all_paths = sorted([os.path.split(file)[1][:-5] for file in some_validated])  # Puros nombres sin extension
except OSError as e:
    logger.exception(f"ERROR VALIDANDO BIBLIOTECAS: {e}", exc_info=True)
    sys.exit()

last_item = len(requeriments_list) - 1
try:
    for i, requierment in enumerate(requeriments_list):
        splited = all_paths[i].split("-")
        name, founded = splited[0], splited[1]
        if i == last_item:
            name = name + python_base

        version_flag = version_call(founded, i)

        if not version_flag or version_flag != requierment:
            raise NameError(f"'{name}' NO COINCIDE LA VERSIÓN: '{founded}'")

        logger.debug(f"'{name.upper()}', version: '{founded}': flag:'{version_flag}'")

        if 3 > i or i > 4:
            bin_path = os.path.join(CONDA_PREFIX, "bin", name)
            if not os.path.exists(bin_path):
                raise FileNotFoundError(f"BINARIO INEXISTENTE: {bin_path}")

            if name == "ninja":
                cacheVariables.update({"CMAKE_MAKE_PROGRAM": bin_path})
            if i == last_item:
                cacheVariables.update({"Python3_EXECUTABLE": bin_path})

            logger.debug(f"'bin path: {bin_path}")
        continue

except OSError as e:
    logger.exception(f"ERROR VALIDADNDO ARCHIVOS: {e}", exc_info=True)
    sys.exit()

num_requeriments = len(_libs_ver)
num_found = len(all_paths)

if num_found != num_requeriments:
    logger.error(f"NO SE ENCONTRARON LAS LIBRERIAS MÍNIMAS PARA EL PIPELINE: {num_requeriments} | {num_found}")
    sys.exit()

file_handler.save_json(*conda_paths, results=digital_presets, mode="w", ident=2)
logger.debug(f"digital: {digital_presets}")
logger.info(f"CONFIG ENV TERMINADO: {time.perf_counter() - time0:.6f}'s")
