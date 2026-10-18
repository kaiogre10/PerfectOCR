import sys
from services.system_service import cleanup_project
from setuptools import setup, Extension
from Cython.Build import cythonize # type: ignore
from typing import Dict, Any
from utils.paths import build_from_dir

def build_extensions(config: Dict[str, Any]):
    utils_file = config.get("comp_funcs_file", "")
    del_files, del_paths, del_build = config["del_files"]
    del_files_path = build_from_dir(del_files, parent=del_paths, with_extensions=True)
    print(f"DEL FILES: {del_files_path}")
    cleanup_project(execp=([], del_files_path))

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

    cleanup_project(execp=(del_build, []))