# PerfectOCR/management/workflow_manager.py
import os
import logging
from typing import List, Dict, Any, Set

logger = logging.getLogger(__name__)

class WorkFlowBuilder:
    """
    Director de Logística: Planifica, cuenta y reporta a Main.
    HIPER-ESPECIALIZADO en: contar imágenes, decidir modo, generar reporte. N procesa imágenes - solo planifica.
    """
    def __init__(self, builder_config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.valid_extensions = builder_config['valid_extensions']
        self.input_paths: List[str] = builder_config['input_dirs']
        self.images_names = set(builder_config['images_names'])
        
    def _extract_valid_image_paths(self, input_folder: str) -> List[Dict[str, Any]]:
        """Extrae lista de rutas y nombres de imágenes válidas de forma recursiva."""
        try:
            image_info: List[Dict[str, str]] = []
            for root, _, files in os.walk(input_folder):
                for filename in files:
                    if filename.lower().endswith(self.valid_extensions):
                        full_path = os.path.join(root, filename)
                        image_name, _ = os.path.splitext(filename)

                        if self.images_names and image_name not in self.images_names:
                            continue

                        image_info.append({
                            "full_path": full_path,
                            "name": image_name
                        })

            if not image_info:
                logger.error(f"No se encontraron imágenes con extensiones válidas en {os.path.join(self.project_root, input_folder)}")
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
        if not self.input_paths:
            return {}

        for path in self.input_paths:
            if os.path.isdir(path):
                imgs = self._extract_valid_image_paths(path)
                for img in imgs:
                    if img["name"] not in seen_names:
                        image_info.append(img)
                        seen_names.add(img["name"])

            elif os.path.isfile(path) and path.lower().endswith(self.valid_extensions):
                base = os.path.basename(path)
                name = os.path.splitext(base)[0]
                if name not in seen_names:
                    image_info.append({"full_path": path,"name": name})
                    seen_names.add(name)
            else:
                logger.error("SIN RUTAS VÁLIDAS, VERIFICAR NOMBRE Y EXTENSIONES DE LAS IMÁGENES REQUERIDAS")        
                return {}
                
        if not image_info:
            return {}
            
        return  {
            "image_info": image_info,
        }