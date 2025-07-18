# PerfectOCR/cli/main.py
import os
import sys
import typer
import logging
from pathlib import Path
from typing import Optional

# Añadir el directorio padre al path para imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from managment.workflow_manager import WorkflowManager

app = typer.Typer(help="PerfectOCR - Sistema de OCR optimizado")

@app.command()
def run(
    input_path: str = typer.Argument(..., help="Carpeta de imágenes o imagen individual"),
    output_dir: str = typer.Option("./output", "--output", "-o", help="Directorio de salida"),
    config: str = typer.Option("./config/master_config.yaml", "--config", "-c", help="Archivo de configuración"),
    mode: Optional[str] = typer.Option(None, "--mode", "-m", help="Forzar modo: 'interactive' o 'batch'"),
    workers: Optional[int] = typer.Option(None, "--workers", "-w", help="Número de procesos a utilizar en modo lote"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Solo mostrar estimación sin procesar"),
):
    """Ejecuta PerfectOCR en modo automático u optimizado."""
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Validar rutas
    input_path_obj = Path(input_path)
    if not input_path_obj.exists():
        typer.echo(f"Error: La ruta {input_path} no existe", err=True)
        raise typer.Exit(1)
    
    # Obtener lista de imágenes
    valid_extensions = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
    
    if input_path_obj.is_file():
        image_paths = [input_path_obj]
    else:
        image_paths = []
        for ext in valid_extensions:
            image_paths.extend(input_path_obj.glob(ext))
    
    if not image_paths:
        typer.echo("No se encontraron imágenes válidas", err=True)
        raise typer.Exit(1)
    
    # Crear manager con override de workers si se proporcionó
    manager = WorkflowManager(config, max_workers_override=workers)
    
    # Mostrar información
    use_batch = manager.should_use_batch_mode(len(image_paths)) if not mode else (mode == 'batch')
    typer.echo(f"Encontradas {len(image_paths)} imágenes")
    typer.echo(f"Modo seleccionado: {'LOTE' if use_batch else 'INTERACTIVO'}")
    
    if dry_run:
        from core.utils.batch_tools import estimate_processing_time
        estimation = estimate_processing_time(len(image_paths))
        typer.echo(f"Estimación de tiempo: {estimation['parallel_minutes']:.1f} minutos")
        typer.echo(f"Workers que se usarían: {estimation['workers']}")
        typer.echo(f"Aceleración esperada: {estimation['speedup']:.1f}x")
        return
    
    # Procesar
    try:
        results = manager.process_images(image_paths, output_dir, force_mode=mode, workers_override=workers)
        typer.echo(f"✅ Procesamiento completado: {results['processed']} imágenes en modo {results['mode']}")
    except Exception as e:
        typer.echo(f"❌ Error durante el procesamiento: {e}", err=True)
        raise typer.Exit(1)

@app.command()
def benchmark(
    input_dir: str = typer.Argument(..., help="Carpeta con imágenes de prueba"),
    config: str = typer.Option("./config/master_config.yaml", "--config", "-c", help="Archivo de configuración"),
):
    """Compara rendimiento entre modo interactivo y lote."""
    import time
    
    input_path = Path(input_dir)
    image_paths = list(input_path.glob("*.png"))[:10]  # Máximo 10 para benchmark
    
    if len(image_paths) < 2:
        typer.echo("Se necesitan al menos 2 imágenes para benchmark", err=True)
        return
    
    manager = WorkflowManager(config)
    
    # Benchmark modo interactivo
    typer.echo("🔄 Benchmarking modo interactivo...")
    start_time = time.time()
    results_interactive = manager.process_images(image_paths, "./output", force_mode='interactive')
    time_interactive = time.time() - start_time
    
    # Benchmark modo lote
    typer.echo("🔄 Benchmarking modo lote...")
    start_time = time.time()
    results_batch = manager.process_images(image_paths, "./output", force_mode='batch')
    time_batch = time.time() - start_time
    
    # Mostrar resultados
    typer.echo("\n📊 Resultados del benchmark:")
    typer.echo(f"Modo interactivo: {time_interactive:.1f}s ({time_interactive/len(image_paths):.1f}s por imagen)")
    typer.echo(f"Modo lote: {time_batch:.1f}s ({time_batch/len(image_paths):.1f}s por imagen)")
    typer.echo(f"Aceleración real: {time_interactive/time_batch:.2f}x")

if __name__ == "__main__":
    app()