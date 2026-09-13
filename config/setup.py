import sys
import os
from setuptools import setup, Extension
import numpy as np
from Cython.Build import cythonize
from services.system_service import cleanup_project
from typing import Dict, Any

def build_extensions(PROJECT_ROOT: str, config: Dict[str, Any]):
    utils_file = config.get("comp_funcs_file", "")
    image_file = config.get("comp_services_file", "")
    prune_workspace(PROJECT_ROOT, image_file, utils_file)
    
    comp_utils_name = config.get("comp_funcs_name", "")
    compiled_services_path = config.get("comp_services_path", "")
    imge_name = "utils.compiled_services.image"
    
    libs_dir = config["libs_path"]
    components_paths = config["components_paths"]
    
    components_paths.append(np.get_include())
    
    components_name = config["components"]

    extensions = [
        Extension(
            name=comp_utils_name,
            sources=[utils_file],
        ),
        Extension(
            name=imge_name,
            sources=[image_file],
            include_dirs=components_paths,
            library_dirs=libs_dir,
            libraries=components_name + [
                "opencv_core",
                "opencv_imgproc",
                "opencv_imgcodecs"
            ],
            language="c++",
            extra_compile_args=[
                "-std=c++20",
                "-mavx2",
                "-mfma",
            ],
        )
    ]

    command = config["compile_command"]
    old_argv = sys.argv
    sys.argv = command
    try:
        setup(
            ext_modules=cythonize(
                extensions,
                compiler_directives={"language_level": "3"},
                include_path=[compiled_services_path]
            ),
        )
    finally:
        sys.argv = old_argv

    prune_workspace(PROJECT_ROOT, "", "")
    
def prune_workspace(PROJECT_ROOT: str, image_file: str, utils_file: str):
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
    aditional_dirs = [os.path.join(PROJECT_ROOT, "build")]
    
    cleanup_project(specific_files=specific_files, aditional_dirs=aditional_dirs)
