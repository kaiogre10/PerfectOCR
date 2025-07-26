import os
from pathlib import Path

def count_lines_in_file(filepath: Path) -> dict:
    """
    Cuenta las líneas de código, comentarios y líneas en blanco de un archivo.
    """
    stats = {'code': 0, 'comment': 0, 'blank': 0, 'total': 0}
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                stats['total'] += 1
                stripped_line = line.strip()
                if not stripped_line:
                    stats['blank'] += 1
                elif stripped_line.startswith('#'):
                    stats['comment'] += 1
                else:
                    stats['code'] += 1
    except Exception as e:
        print(f"Error al leer {filepath}: {e}")
    return stats

def analyze_project(root_dir: Path):
    """Analiza el proyecto, ignorando directorios y archivos no relevantes."""
    ignored_docs = {'metrics.py'}
    
    # Directorios a ignorar
    ignore_dirs = {
        '__pycache__', 'output', 'input', 'models', '.git', '.vscode', 'data', '.txt', 'data'
    }
    # Extensiones de archivo a incluir
    include_exts = {'.py', '.yaml', '.yml'}

    all_stats = {}
    total_summary = {'code': 0, 'comment': 0, 'blank': 0, 'total': 0, 'files': 0}

    for path in root_dir.rglob('*'):
        if path.is_file():
            # Ignorar si el archivo está en un directorio no deseado
            if any(ignored in path.parts for ignored in ignore_dirs and ignored_docs):
                continue
            
            if path.suffix in include_exts:
                file_stats = count_lines_in_file(path)
                relative_path = path.relative_to(root_dir)
                all_stats[str(relative_path)] = file_stats
                
                # Sumar al total
                for key in ['code', 'comment', 'blank', 'total']:
                    total_summary[key] += file_stats[key]
                total_summary['files'] += 1

    # Imprimir resultados con mejor formato
    print("ANÁLISIS DE LÍNEAS DE CÓDIGO -  PerfectOCR")
    print("="*100)
    
    # Encabezado de tabla mejorado
    header = f"{'ARCHIVO':<60} {'CÓDIGO':>8} {'COMENTARIOS':>12} {'BLANCOS':>8} {'TOTAL':>8}"
    print(header)
    print("="*100)

    # Ordenar por líneas de código descendente
    sorted_stats = sorted(all_stats.items(), key=lambda item: item[1]['code'], reverse=True)

    for filepath, stats in sorted_stats:
        display_path = filepath if len(filepath) <= 60 else f"...{filepath[-57:]}"
        print(f"{display_path:<60} {stats['code']:>8} {stats['comment']:>12} {stats['blank']:>8} {stats['total']:>8}")

    print("="*100)
    print("RESUMEN DEL PROYECTO:")
    print(f"Archivos analizados: {total_summary['files']:,}")
    print(f"Líneas de código (SLOC): {total_summary['code']:,}")
    print(f"Líneas de comentarios: {total_summary['comment']:,}")
    print(f"Líneas en blanco: {total_summary['blank']:,}")
    print(f"Total de líneas: {total_summary['total']:,}")

if __name__ == "__main__":
    project_root = Path(__file__).parent
    analyze_project(project_root)