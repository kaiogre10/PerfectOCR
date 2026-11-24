# safe_temp/config_env.py
import os
import sys

def _run():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    import time
    from services import log_service
    from safe_temp import env_assets
    from utils import paths, file_handler
    logger = log_service.log_temp_process(project_root)

    try:
        time0 = time.perf_counter()
        # if os.path.exists(os.path.join(project_root, "build", "compile_commands.json")):
        #     logger.warning("PROYECTO PREVIAMENTE COMPILADO Y CONFIGUTANDO, ABORTANDO")
        #     del logger, project_root
        #     return

        conda_paths = paths.build_from_dir(objetives=["CondaPaths.json"], search_root=project_root, parent=["cmake"])
        if not conda_paths:
            logger.warning("NO HAY CONDA PATHS")
            del logger
            return

        env_path = os.path.join(*conda_paths)
        digital_presets = file_handler.load_json(env_path, mode="r")
        dict_items = digital_presets["configurePresets"][0]
        cachevariables = dict_items.get("cacheVariables", {})
        
        ONEAPI_ROOT = os.environ.get("ONEAPI_ROOT", "")
        if not ONEAPI_ROOT:
            pass

        elif not os.path.exists(ONEAPI_ROOT):
            logger.debug(f"NO SE ENCONTRO EL ONEAPI_ROOT: {ONEAPI_ROOT}, POR LO QUE NO ESTARA DISPONIBLE LA COMPILACION OPTIMIZADA")
            del cachevariables["ONEAPI_ROOT"]
        else:
            logger.debug(f"ONEAPI ROOT: {ONEAPI_ROOT}")    
            cachevariables.update({"ONEAPI_ROOT": ONEAPI_ROOT})
        
        python_path = sys.executable
        
        envs_path = os.path.dirname(os.path.normpath(python_path))
        if os.path.exists(os.path.join(envs_path, "envs")):
            logger.error("NO SE ACTIVO EL INTERPRETE DEL CONDA ENV, REVISAR CONDA")
            return
        
        if env_assets.WIN:
            validation_root = os.path.join(envs_path, "conda-meta")
            bins_path = os.path.join(envs_path, "Library")
            CONDA_PREFIX = envs_path
            CMAKE_PREFIX_PATH = os.path.join(CONDA_PREFIX, "Library")
        
        else:
            CONDA_PREFIX = os.path.dirname(envs_path)
            validation_root = os.path.join(CONDA_PREFIX, "conda-meta")
            bins_path = CONDA_PREFIX
            CMAKE_PREFIX_PATH = CONDA_PREFIX

        logger.debug(f"SO: {env_assets.WIN}")
        logger.debug(f"ENVS PATH: {CONDA_PREFIX}")
        
        cachevariables.update({
            "CONDA_ENV": CONDA_PREFIX,
            "CONDA_PREFIX": CONDA_PREFIX,
            "Python3_ROOT_DIR": CONDA_PREFIX,
            "CMAKE_PREFIX_PATH": CMAKE_PREFIX_PATH,
        })

        if not os.path.exists(validation_root):
            logger.error(f"NO EXXISTE CARPETA DE PARA VALIDACION: {validation_root}")
            return
        
        _libs_ver = env_assets.libs_ver
        num_requeriments = len(_libs_ver)
        
        try:
            some_validated = paths.build_from_dir(objetives=_libs_ver, search_root=validation_root, kind_search="prefix")
            logger.debug(f"VALIDATED:\n{some_validated}")
            all_paths = sorted([os.path.split(file)[1][:-5] for file in some_validated if os.path.exists(file)])  # Puros nombres sin extension
            logger.debug(f"ALL PATHS:\n{some_validated}")
        except OSError as e:
            logger.exception(f"ERROR VALIDANDO BIBLIOTECAS: {e}", exc_info=True)
            return

        if len(all_paths) != num_requeriments:
            logger.error(f"NO SE ENCONTRARON LAS LIBRERIAS MÍNIMAS PARA EL PIPELINE: {num_requeriments} | {len(all_paths)}")
            return
        
        last_item = num_requeriments - 1
        _requeriments_list= env_assets.requeriments_list
        logger.debug(f"REQ LIST: {_requeriments_list}")
        try:
            for i, requierment in enumerate(_requeriments_list):
                splited = all_paths[i].split("-")
                name, ver_founded = splited[0], splited[1]
                logger.debug(f"BEFORE REQUERIMENT: '{requierment}', NAME-FOUNDED: '{name}', '{ver_founded}'")
                version_flag = env_assets.version_call(ver_founded, i)

                if not version_flag or version_flag != requierment:
                    raise NameError(f"'{name}'-'{ver_founded}' NO COINCIDEN FLAGS: '{requierment}' != '{version_flag}'")

                if i == last_item:
                    if not env_assets.WIN:
                        name = name + env_assets.python_base
                    else:
                        name = f"{name}.exe"
                                
                bin_path = os.path.join(bins_path, "bin", name)
                if i < 1:
                    if not os.path.exists(bin_path):
                        raise FileNotFoundError(f"BINARIO INEXISTENTE: {bin_path}")
                    continue
                    
                elif name.startswith("conda"):
                    if env_assets.WIN:
                        bin_path = os.path.join(envs_path, "Scripts", name)
                        
                    if not os.path.exists(bin_path):
                        raise FileNotFoundError(f"BINARIO INEXISTENTE: {bin_path}")

                    cachevariables.update({"CONDA_EXE": bin_path if not "\\" in bin_path else bin_path.replace("\\", "/")})
                    continue
                
                elif name.startswith("ninja"):
                    if env_assets.WIN:
                        name = name.split(".")[0]
                        
                    dict_items.update({"generator": name.capitalize()})
                        
                    if not os.path.exists(bin_path):
                        raise FileNotFoundError(f"BINARIO INEXISTENTE: {bin_path}")
                    cachevariables.update({"CMAKE_MAKE_PROGRAM": bin_path if not "\\" in bin_path else bin_path.replace("\\", "/")})
                    continue
                    
                elif i == last_item:
                    bin_path = os.path.join(envs_path, name)
                    if not os.path.exists(bin_path):
                        raise FileNotFoundError(f"BINARIO INEXISTENTE: {bin_path}")
                    cachevariables.update({"Python3_EXECUTABLE": bin_path if not "\\" in bin_path else bin_path.replace("\\", "/")})
                    continue
                else:
                    continue
        
        except OSError as e:
            logger.exception(f"ERROR VALIDADNDO ARCHIVOS: {e}", exc_info=True)
            return
        try:
            file_handler.save_json(env_path, results=digital_presets, mode="w", ident=2)
        except IOError as e:
            del logger, log_service, file_handler, cachevariables, dict_items, digital_presets
            raise OSError(f"ERROR: {e}")
        logger.warning(f"CONFIG ENV TERMINADO: {time.perf_counter() - time0:.6f}'s:\n"f"{digital_presets}")
        del logger, log_service, file_handler, cachevariables, dict_items, digital_presets
    except Exception as e:
        del logger, log_service, file_handler, cachevariables, dict_items, digital_presets
        raise Exception(f"eerror en config env: {e}")
_run()
del _run