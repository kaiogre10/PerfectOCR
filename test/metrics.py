# PerfectOCR/test/metrics.py
from pathlib import Path
from typing import Dict, Any, Set
import logging

logger = logging.getLogger(__name__)

def count_lines_in_file(filepath: Path) -> Dict[str, int]:
    """
    Cuenta las líneas de código, comentarios y líneas en blanco de un archivo.
    """
    stats: Dict[str, int] = {'code': 0, 'comment': 0, 'blank': 0, 'total': 0, "functions": 0}
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                stats['total'] += 1
                stripped_line = line.strip()
                if not stripped_line:
                    stats['blank'] += 1
                elif stripped_line.startswith('#'):
                    stats['comment'] += 1
                elif stripped_line.startswith(('def')):
                    stats['functions'] += 1
                    stats['code'] += 1
                else:
                    stats['code'] += 1
    except EncodingWarning as e:
        print(f"Error al leer {filepath}: {e}")
    return stats

def analyze_project(root_dir: Path):
    """Analiza el proyecto, ignorando directorios y archivos no relevantes."""
    ignored_docs = {'metrics.py', "__init__.py"}

    # Directorios a ignorar
    ignore_dirs: Set[str] = {
        '__pycache__', 'output', 'input', 'models', '.git', '.vscode', 'data', '.txt', "data_base", "test",
        ".vscode", "input", "tools", "documentation"
    }
    # Extensiones de archivo a incluir
    include_exts: Set[str] = {'.py', '.yaml'}
    exclude_files: Set[str] = {'.txt', ".md", ".env", ".json", ".png", ".pkl"}
    total_summary: Dict[str, int] = {'code': 0, 'comment': 0, 'blank': 0, 'total': 0, "functions": 0, 'files': 0, 'python': 0, 'yaml': 0,}
    all_stats: Dict[str, Any] = {}
    qty_py = 0
    qty_yml = 0
    for path in root_dir.rglob('*'):
        if path.is_file():
            # Ignorar si el archivo está en un directorio no deseado
            if any(ignored in path.parts for ignored in ignore_dirs):
                continue

            if any(ignored in ignored_docs for ignored in path.parts):
                continue

            if path.suffix in exclude_files:
                continue

            elif path.suffix in include_exts:
                file_stats= count_lines_in_file(path)
                relative_path = path.relative_to(root_dir)
                all_stats[str(relative_path)] = file_stats

                # Sumar al total

                for key in {'code', 'comment', 'blank', 'total', 'functions'}:
                    total_summary[key] += file_stats[key]
                total_summary['files'] += 1
                if path.suffix == ".yaml":
                    total_summary['yaml'] += 1
                if path.suffix == ".py":
                    total_summary['python'] += 1
    print("="*100)
    header = f"{'ANÁLISIS DE LÍNEAS DE CÓDIGO':<60} {'CÓDIGO':>8} {'COMENTARIOS':>12} {'BLANCOS':>8} {'TOTAL':>8}"
    print(header)
    print("="*100)

    sorted_stats = sorted(all_stats.items(), key=lambda item: item[1]['code'], reverse=True)
    for filepath, stats in sorted_stats:
        display_path = filepath if len(filepath) <= 60 else f"...{filepath[-57:]}"
        print(f"{display_path:<60} {stats['code']:>8} {stats['comment']:>12} {stats['blank']:>8} {stats['total']:>8}")

    print("RESUMEN DEL PROYECTO:")
    print(f"Archivos analizados: {total_summary['files']:,}, Python: {total_summary['python']:,}, Yaml: {total_summary['yaml']:,}")
    print(f"Cantidad de funciones: {total_summary['functions']:,}")
    print(f"Líneas de código (SLOC): {total_summary['code']:,}")
    print(f"Líneas de comentarios: {total_summary['comment']:,}")
    print(f"Líneas en blanco: {total_summary['blank']:,}")
    print(f"Total de líneas: {total_summary['total']:,}")

try:
    if __name__ == "__main__":
        project_root = Path(__file__).parent.parent
        analyze_project(project_root)
except Exception as e:
    logger.error(f"ERROR: '{e}'", exc_info=True)
