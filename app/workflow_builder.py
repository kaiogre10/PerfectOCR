# PerfectOCR/management/workflow_manager.py
import os
import logging
from typing import List, Dict, Any, Tuple, Set

logger = logging.getLogger(__name__)

class WorkFlowBuilder:
    """
    Director de Logística: Planifica, cuenta y reporta a Main.
    HIPER-ESPECIALIZADO en: contar imágenes, decidir modo, generar reporte. N procesa imágenes - solo planifica.
    """
    def __init__(self, builder_config: Dict[str, Any], project_root: str, input_paths: List[str] | str):
        self.project_root = project_root
        self.valid_extensions = builder_config['valid_image_extensions']
        self.input_paths = input_paths
        
    def _extract_valid_image_paths(self, input_folder: str, valid_extensions: Tuple[str, ...]) -> List[Dict[str, Any]]:
        """Extrae lista de rutas y nombres de imágenes válidas de forma recursiva."""
        image_info: List[Dict[str, str]] = []
        try:
            for root, _, files in os.walk(input_folder):
                for filename in files:
                    if filename.lower().endswith(valid_extensions):
                        full_path = os.path.join(root, filename)
                        image_name, image_extension = os.path.splitext(filename)

                        # Obtener ruta relativa desde input_folder para mejor organización
                        relative_path = os.path.relpath(root, input_folder)
                        if relative_path == ".":
                            relative_path = ""
                        
                        image_info.append({
                            "full_path": full_path,
                            "name": image_name,
                            "extension": image_extension,
                            "relative_folder": relative_path,
                        })

            if image_info:
                logger.debug(f"Encontradas {len(image_info)} imágenes en {input_folder} y subcarpetas")
                # Mostrar estructura de carpetas encontradas
                folders_found = set(img["relative_folder"] for img in image_info if img["relative_folder"])
                if folders_found:
                    logger.debug(f"Subcarpetas con imágenes: {sorted(folders_found)}")
            else:
                logger.warning(f"No se encontraron imágenes con extensiones {valid_extensions} en {input_folder}")
                return []
                
            return image_info
            
        except Exception as e:
            logger.error(f"Error validando rutas: {e}", exc_info=True)
        return []

    def count_and_plan(self) -> Dict[str, Any]:
        """
        PLANIFICA el procesamiento: cuenta imágenes y decide estrategia.
        REPORTA a Main: cuántos builders crear y qué modo usar.
        """
        image_info: List[Dict[str, Any]] = []
        seen_names: Set[str] = set()
        try:
            if self.input_paths:
                for path in self.input_paths:
                    if os.path.isdir(path):
                        imgs = self._extract_valid_image_paths(path, self.valid_extensions)
                        for img in imgs:
                            if img["name"] not in seen_names:
                                image_info.append(img)
                                seen_names.add(img["name"])
                    elif os.path.isfile(path) and path.lower().endswith(self.valid_extensions):
                        base = os.path.basename(path)
                        name = os.path.splitext(base)[0]
                        if name not in seen_names:
                            image_info.append({
                                "path": path,
                                "name": name,
                                "extension": os.path.splitext(base)[1],
                            })
                            seen_names.add(name)

            num_images = len(image_info)
            logger.info(f"Número de imágenes: {num_images}")

            return {
                "total_images": num_images,
                "image_info": image_info,
            }
        
        except Exception as e:
            logger.error(f"Error contando: {e}", exc_info=True)
        return {}