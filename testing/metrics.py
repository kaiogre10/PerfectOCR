import os
import sys
root = os.path.join("../")
project_root = sys.path.insert(0, root)
import logging
from typing import Dict, List, Set
from domain.class_models import Metrics

logger = logging.getLogger(__name__)

_include_exts: Set[str] = {'.py', ".c", ".hpp", ".cpp", ".h", ".env", ".pyi", ".pyx", ".pxd", ".sql", ".txt", ".hpp", ".yaml"}
ignored_docs = {'metrics.py', "__init__.py"}
ignore_dirs = {'__pycache__', 'output', 'input', 'models', '.git', '.vscode', 'data', ".vs", "build", "bin", "CmakeFiles", "src", "opencv_dep"}
filtred_dirs = ("cmake", "directory", "reply", "directory-")
exclude_files = {'.txt', ".md", ".env", ".png", ".pkl", "opencv_dep"}


def count_lines_in_file(filepath: str) -> Dict[str, int]:
    stats: Dict[str, int] = {Metrics.code: 0, Metrics.comment: 0, Metrics.blank: 0, Metrics.total: 0, Metrics.functions: 0}
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                stats[Metrics.total] += 1
                stripped = line.strip()
                if not stripped:
                    stats[Metrics.blank] += 1
                elif stripped.startswith('#'):
                    stats[Metrics.comment] += 1
                elif stripped.startswith('def '):
                    stats[Metrics.functions] += 1
                    stats[Metrics.code] += 1
                else:
                    stats[Metrics.code] += 1
    except Exception as e:
        print(f"Error al leer {filepath}: {e}")
    return stats

def get_project_root() -> str:
    current_dir = os.path.abspath(os.path.dirname(__file__))
    while True:
        if os.path.exists(os.path.join(current_dir, '.git')) or os.path.exists(os.path.join(current_dir, 'requirements.txt')):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            return os.path.abspath(os.path.dirname(__file__)) # Fallback seguro
        current_dir = parent_dir

def analyze_project(project_root: str):
    total_summary: Dict[str, int] = {Metrics.code: 0, Metrics.comment: 0, Metrics.blank: 0, Metrics.total: 0, Metrics.functions: 0, Metrics.files: 0, Metrics.python: 0, Metrics.yaml: 0, Metrics.cplusplus: 0, Metrics.cython: 0, Metrics.sql: 0}
    all_stats = {}
    
    # Reemplazo de rglob usando os.walk
    for root, dirs, files in os.walk(project_root):
        # Filtrar directorios ignorados in-place para que os.walk no entre en ellos
        dirs[:] = [d for d in dirs if d not in ignore_dirs or d.startswith(filtred_dirs)]
        
        for file in files:
            if file in ignored_docs:
                continue
                
            ext = os.path.splitext(file)[1]
            if ext in exclude_files or ext not in _include_exts:
                continue
                
            full_path = os.path.join(root, file)
            file_stats = count_lines_in_file(full_path)
            
            # Obtener ruta relativa nativa con os.path
            relative_path = os.path.relpath(full_path, project_root)
            all_stats[relative_path] = file_stats

            for key in {Metrics.code, Metrics.comment, Metrics.blank, Metrics.total, Metrics.functions}:
                total_summary[key] += file_stats[key]
            
            total_summary[Metrics.files] += 1
            if ext == ".yaml": total_summary[Metrics.yaml] += 1
            elif ext == ".py": total_summary[Metrics.python] += 1
            elif ext in {".cpp", ".h"}: total_summary[Metrics.cplusplus] += 1
            elif ext in {".pyx", ".pyi"}: total_summary[Metrics.cython] += 1
            elif ext == ".sql": total_summary[Metrics.sql] += 1

    print("="*100)
    print(f"{'ANÁLISIS DE LÍNEAS DE CÓDIGO':<60} {'CÓDIGO':>8} {'COMENTARIOS':>12} {'BLANCOS':>8} {Metrics.total:>8}")
    print("="*100)

    sorted_stats: List[str] = sorted(all_stats.items(), key=lambda item: item[1][Metrics.code], reverse=True)
    for filepath, stats in sorted_stats:
        display_path = filepath if len(filepath) <= 60 else f"...{filepath[-57:]}"
        print(f"{display_path:<60} {stats[Metrics.code]:>8} {stats[Metrics.comment]:>12} {stats[Metrics.blank]:>8} {stats[Metrics.total]:>8}")

    print("\nRESUMEN DEL PROYECTO:")
    print(f"Archivos analizados: {total_summary[Metrics.files]:,}, Python: {total_summary[Metrics.python]:,}, Yaml: {total_summary[Metrics.yaml]:,}, C++: {total_summary[Metrics.cplusplus]:,}, Cython: {total_summary[Metrics.cython]:,}, SQL: {total_summary[Metrics.sql]:,}")
    print(f"Cantidad de funciones: {total_summary[Metrics.functions]:,}")
    print(f"Líneas de código (SLOC): {total_summary[Metrics.code]:,}")
    print(f"Líneas de comentarios: {total_summary[Metrics.comment]:,}")
    print(f"Líneas en blanco: {total_summary[Metrics.blank]:,}")
    print(f"Total de líneas: {total_summary[Metrics.total]:,}")

if __name__ == "__main__":
    try:
        project_root = get_project_root()
        analyze_project(project_root)
    except Exception as e:
        logger.error(f"ERROR: '{e}'", exc_info=True)