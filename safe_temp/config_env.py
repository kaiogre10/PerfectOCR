import os
import sys
import time
from services import log_service
from utils import paths, file_handler
import importlib

def _run():
    try:
        time0 = time.perf_counter()
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        logger = log_service.log_temp_process(project_root)
        importlib.import_module("env_assets")

        if os.path.exists(os.path.join(project_root, "build", "COMPILE_COMMANDS.json")):
            logger.info("PROYECTO PREVIAMENTE COMPILADO Y CONFIGUTANDO, ABORTANDO")
            del logger, project_root
            return
        
        conda_paths = paths.build_from_dir(objetives=["CondaPaths.json"], search_root=project_root)
        if not conda_paths:
            logger.warning("NO HAY CONDA PATHS")
            del logger, project_root
            return
        
        digital_presets = file_handler.load_json(*conda_paths, mode="r")
        dict_items = digital_presets["configurePresets"][0]
        cachevariables = dict_items.get("cacheVariables", {})
        
        ONEAPI_ROOT = os.environ.get("ONEAPI_ROOT", "")
        if not os.path.exists(ONEAPI_ROOT):
            print(f"NO SE ENCONTRO LA EL ONEAPIROOT: {ONEAPI_ROOT}, POR LO QUE NO ESTARA DISPONIBLE LA COMPILACION OPTIMIZADA")
        else:
            print(f"ONEAPI ROOT: {ONEAPI_ROOT}")
            
        cachevariables.update({"ONEAPI_ROOT": ONEAPI_ROOT})
        
        python_path = sys.executable
        
        envs_path = os.path.dirname(os.path.normpath(python_path))
        if os.path.exists(os.path.join(envs_path, "envs")):
            logger.error("NO SE ACTIVO EL INTERPRETE DEL CONDA ENV, REVISAR CONDA")
            return
        
        if env_assets.WIN:
            validation_root = os.path.join(envs_path, "conda-meta")
            libs_path = os.path.join(envs_path, "Library")
            CONDA_PREFIX = envs_path
            CMAKE_PREFIX_PATH = os.path.join(CONDA_PREFIX, "Library")
        
        else:
            CONDA_PREFIX = os.path.dirname(envs_path)
            validation_root = os.path.join(CONDA_PREFIX, "conda-meta")
            libs_path = CONDA_PREFIX
            CMAKE_PREFIX_PATH = CONDA_PREFIX
        
        logger.debug(f"ENVS PATH: {envs_path}")
        
        cachevariables.update({
            "CONDA_ENV": CONDA_PREFIX,
            "CONDA_PREFIX": CONDA_PREFIX,
            "Python3_ROOT_DIR": CONDA_PREFIX,
            "CMAKE_PREFIX_PATH": CMAKE_PREFIX_PATH,
        })
        
        if not os.path.exists(validation_root):
            logger.error("NO EXXISTE CARPETA DE PARA VALIDACION")
            return
        
        _libs_ver = env_assets.libs_ver
        num_requeriments = len(_libs_ver)
        
        try:
            some_validated = paths.build_from_dir(objetives=_libs_ver, search_root=validation_root, kind_search="prefix")
            all_paths = sorted([os.path.split(file)[1][:-5] for file in some_validated if os.path.exists(file)])  # Puros nombres sin extension
        except OSError as e:
            logger.exception(f"ERROR VALIDANDO BIBLIOTECAS: {e}", exc_info=True)
            return
            
        if len(all_paths) != num_requeriments:
            logger.error(f"NO SE ENCONTRARON LAS LIBRERIAS MÍNIMAS PARA EL PIPELINE: {num_requeriments} | {len(all_paths)}")
            return
        
        last_item = num_requeriments - 1
        
        try:
            for i, requierment in enumerate(env_assets.requeriments_list):
                splited = all_paths[i].split("-")
                name, ver_founded = splited[0], splited[1]
                
                if i == last_item and not env_assets.WIN:
                    name = name + env_assets.python_base
                else:
                    name = f"{name}.exe"
                    
                version_flag = env_assets.version_call(ver_founded, i)
                
                if not version_flag or version_flag != requierment:
                    raise NameError(f"'{name}' NO COINCIDEN FLAGS: {requierment} != {version_flag} VERSIÓN: '{ver_founded}'")
        
                # logger.debug(f"'{name.upper()}', version: '{founded}': flag:'{version_flag}'")
                bin_path = os.path.join(libs_path, "bin", name)
                if i < 1:
                    if not os.path.exists(bin_path):
                        raise FileNotFoundError(f"BINARIO INEXISTENTE: {bin_path}")
                    continue
                    
                elif name.startswith("conda"):
                    if env_assets.WIN:
                        bin_path = os.path.join(envs_path, "Scripts", name)
                        
                    if not os.path.exists(bin_path):
                        raise FileNotFoundError(f"BINARIO INEXISTENTE: {bin_path}")

                    cachevariables.update({"CONDA_EXE": bin_path})
                    continue
                
                elif name.startswith("ninja"):
                    if env_assets.WIN:
                        name = name.split(".")[0]
                        
                    dict_items.update({"generator": name.capitalize()})
                        
                    if not os.path.exists(bin_path):
                        raise FileNotFoundError(f"BINARIO INEXISTENTE: {bin_path}")
                    cachevariables.update({"CMAKE_MAKE_PROGRAM": bin_path})
                    continue
                    
                elif i == last_item:
                    bin_path = os.path.join(envs_path, name)
                    if not os.path.exists(bin_path):
                        raise FileNotFoundError(f"BINARIO INEXISTENTE: {bin_path}")
                    cachevariables.update({"Python3_EXECUTABLE": bin_path})
                    continue
                else:
                    continue
        
        except OSError as e:
            logger.exception(f"ERROR VALIDADNDO ARCHIVOS: {e}", exc_info=True)
            return
        
        file_handler.save_json(*conda_paths, results=digital_presets, mode="w", ident=2)
        logger.info(f"CONFIG ENV TERMINADO: {time.perf_counter() - time0:.6f}'s")
        del logger, log_service, project_root, file_handler, ONEAPI_ROOT, cachevariables, dict_items, digital_presets
    except Exception:
        return
_run()
del _run