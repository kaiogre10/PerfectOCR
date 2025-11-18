# PerfectOCR/management/workflow_manager.py
import os
import logging
from PIL import Image
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class WorkFlowBuilder:
    """
    Director de Logística: Planifica, cuenta y reporta a Main.
    HIPER-ESPECIALIZADO en: contar imágenes, decidir modo, generar reporte.
    NO procesa imágenes - solo planifica.
    """
    def __init__(self, config: Dict[str, Any], project_root: str, input_paths: List[str] | str):
        self.project_root = project_root
        self.utils_config = config.get("utils_config", {})
        self.dpi_range: List[int] = self.utils_config["dpi_range"]
        self.builder_config = config.get("processing", {})
        self.small_batch_limit: int = self.builder_config.get('small_batch_limit')
        self.valid_extensions = self.builder_config['valid_image_extensions']
        self.input_paths = input_paths
        
    def _extract_valid_image_paths(self, input_folder: str, valid_extensions: Tuple[str, ...]) -> List[Dict[str, Any]]:
        """Extrae lista de rutas y nombres de imágenes válidas de forma recursiva."""
        image_info: List[Dict[str, str]] = []
        for root, dirs, files in os.walk(input_folder): # type: ignore
            for filename in files:
                if filename.lower().endswith(valid_extensions):
                    full_path = os.path.join(root, filename)
                    image_name = os.path.splitext(filename)[0]
                    image_extension = os.path.splitext(filename)[1]

                    with Image.open(full_path) as img:
                        dpi_aprox = img.info["dpi"][1]
                        dpi_cal = self.closest_int(dpi_aprox, self.dpi_range)

                    # Obtener ruta relativa desde input_folder para mejor organización
                    relative_path = os.path.relpath(root, input_folder)
                    if relative_path == ".":
                        relative_path = ""
                    
                    image_info.append({
                        "full_path": full_path,
                        "name": image_name,
                        "extension": image_extension,
                        "relative_folder": relative_path,
                        "dpi": dpi_cal
                    })

        if image_info:
            logger.debug(f"Encontradas {len(image_info)} imágenes en {input_folder} y subcarpetas")
            # Mostrar estructura de carpetas encontradas
            folders_found = set(img["relative_folder"] for img in image_info if img["relative_folder"])
            if folders_found:
                logger.debug(f"Subcarpetas con imágenes: {sorted(folders_found)}")
        else:
            logger.warning(f"No se encontraron imágenes con extensiones {valid_extensions} en {input_folder}")
            
        return image_info

    def count_and_plan(self) -> Dict[str, Any]:
        """
        PLANIFICA el procesamiento: cuenta imágenes y decide estrategia.
        REPORTA a Main: cuántos builders crear y qué modo usar.
        """
        # Si recibimos input_paths, expandimos; si no, usamos la carpeta del YAML
        image_info: List[Dict[str, Any]] = []
        if self.input_paths:
            for path in self.input_paths:
                if os.path.isdir(path):
                    image_info.extend(self._extract_valid_image_paths(path, self.valid_extensions))
                elif os.path.isfile(path) and path.lower().endswith(self.valid_extensions):
                    base = os.path.basename(path)
                    image_info.append({
                        "path": path,
                        "name": os.path.splitext(base)[0],
                        "extension": os.path.splitext(base)[1],
                    })

        if not image_info:
            logger.critical("No se encontraron imágenes válidas.")
            return {}

        num_images = len(image_info)
        use_batch = num_images > self.small_batch_limit
        mode = 'batch' if use_batch else 'interactive'
        logging.debug(f"Número de imágenes: {num_images}, modo: {mode}")

        return {
            "total_images": num_images,
            "mode": mode,
            "image_info": image_info,
        }

    def closest_int(self, value: float, candidates: List[int]) -> int:
       return min(candidates, key=lambda x: abs(x - value))