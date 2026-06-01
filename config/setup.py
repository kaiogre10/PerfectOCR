import os
import sys
from setuptools import setup, Extension
from Cython.Build import cythonize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)  # ← Sube a la raíz

# Añade PROJECT_ROOT al path ANTES de importar services
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from services.system_service import cleanup_project_cache

PYX_FILE = os.path.abspath(os.path.join(PROJECT_ROOT, "core", "utils", "compiled_utils", "compiled_funcs.pyx"))
cleanup_project_cache(aditional_files=".c")

extensions = [
    Extension(
        name="core.utils.compiled_utils.compiled_funcs",
        sources=[PYX_FILE],
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    )
)