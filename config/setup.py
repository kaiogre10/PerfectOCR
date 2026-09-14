import os
import sys
from services.system_service import cleanup_project
from setuptools import setup, Extension
import numpy as np
from Cython.Build import cythonize
from typing import Dict, Any

extra_compile_args = [
    "-std=c++20",
    "-mavx2",
    "-mfma",
]


def build_extensions(PROJECT_ROOT: str, config: Dict[str, Any]):
    utils_file = config.get("comp_funcs_file", "")
    image_file = config.get("comp_services_file", "")
    prune_workspace(config["build_path"], image_file, utils_file)
    
    comp_utils_name = config.get("comp_funcs_name", "")
    compiled_services_path = config["comp_services_path"]
    include_dirs = [*config["components_paths"], np.get_include()] # type: ignore
    library_dirs = config["library_dirs"]
    libraries = config["libraries"]
    runtime_library_dirs = config["runtime_library_dirs"]

    extensions = [
        Extension(
            name=comp_utils_name,
            sources=[utils_file],
        ),
        Extension(
            name="utils.compiled_services.image",
            sources=[image_file],
            language="c++",
            include_dirs=include_dirs,  # type: ignore
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args=extra_compile_args,
            runtime_library_dirs=runtime_library_dirs,
        ),
    ]

    command = config["compile_command"]
    old_argv = sys.argv
    sys.argv = command
    try:
        setup(
            name="compiled_services",
            ext_modules=cythonize(
                extensions,
                compiler_directives={"language_level": "3"},
                include_path=[compiled_services_path]
            )
        )
    finally:
        sys.argv = old_argv

    prune_workspace(config["build_path"], "", "")
    
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
