import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from setuptools import setup, Extension
from Cython.Build import cythonize
import services.system_service as system_service

PYX_FILE = os.path.abspath(os.path.join(PROJECT_ROOT, "utils", "compiled_utils", "compiled_funcs.pyx"))

extensions = [
    Extension(
        name="utils.compiled_utils.compiled_funcs",
        sources=[PYX_FILE],
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    )
)

system_service.set_system_config(PROJECT_ROOT, {})
system_service.cleanup_project(aditional_files=".c")