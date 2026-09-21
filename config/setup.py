import os
import sys
from services.system_service import cleanup_project
from setuptools import setup, Extension
from Cython.Build import cythonize # type: ignore
from typing import Dict, Any

def build_extensions(config: Dict[str, Any]):
    utils_file = config.get("comp_funcs_file", "")
    build_path = "build"

    prune_workspace(build_path, "", utils_file)

    comp_funcs_name = config.get("comp_funcs_name", "")

    extensions = [
        Extension(
            name=comp_funcs_name,
            sources=[utils_file],
        )
    ]

    command = config["compile_command"]
    old_argv = sys.argv
    sys.argv = command
    try:
        setup(
            ext_modules=cythonize(
                extensions,
                compiler_directives={"language_level": "3"}
            )
        )
    finally:
        sys.argv = old_argv

    prune_workspace(build_path, "", "")
    
def prune_workspace(build_path: str, image_file: str, utils_file: str):
    if not image_file:
        target_cpp = ""
        if utils_file:
            utils_path = os.path.splitext(utils_file)[0]
            target_c = (utils_path + ".c")
        else:
            target_c = ""
            
    elif not utils_file:
        target_c = ""
        image_path = os.path.splitext(image_file)[0]
        target_cpp = (image_path + ".cpp")
        
    else:
        utils_path = os.path.splitext(utils_file)[0]
        image_path = os.path.splitext(image_file)[0]
        target_cpp = (image_path + ".cpp")
        target_c = (utils_path + ".c")
        
    specific_files = [target_cpp, target_c]
    aditional_dirs = [build_path]
    
    cleanup_project(specific_files=specific_files, aditional_dirs=aditional_dirs)
