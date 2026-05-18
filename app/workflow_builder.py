# PerfectOCR/management/workflow_manager.py
import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class WorkFlowBuilder:
    """
    Director de Logística: Planifica, cuenta y reporta a Main.
    HIPER-ESPECIALIZADO en: contar imágenes, decidir modo, generar reporte. N procesa imágenes - solo planifica.
    """
    def __init__(self, builder_config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.valid_extensions = tuple(ext.lower() for ext in builder_config['valid_extensions'])
        self.input_paths: List[str] = builder_config['input_dirs']
        self.images_names = builder_config['images_names']
        
    def count_and_plan(self) -> Dict[str, Any]:
        """
        PLANIFICA el procesamiento: cuenta imágenes y decide estrategia según las reglas:
        1. Si se especifican `images_names`, se buscan prioritariamente.
        2. Si no, se procesan todas las imágenes en `input_dirs`.
        3. Si se encuentran todos los `images_names` y quedan directorios, se procesan completos.
        """
        if not self.input_paths:
            logger.warning("No se proporcionaron rutas de entrada (input_dirs).")
            return {}

        image_info: List[Dict[str, Any]] = []
        names_to_find = set(self.images_names)
        total_paths = len(self.input_paths)
        
        for i, path in enumerate(self.input_paths):
            if names_to_find:
                files_in_dir = self.get_images_in_dir(path, list(names_to_find))
                if files_in_dir:
                    files_to_remove = set(files_in_dir)
                    if not names_to_find.isdisjoint(files_to_remove):
                        names_to_find.discard(files_to_remove)
                    
                    for file in files_in_dir:
                        full_path = os.path.join(self.project_root, path, file)
                        image_info.append({
                            "full_path": full_path,
                            "name": file
                        })
                        continue
                    
                elif names_to_find:
                    continue
                
            elif total_paths >= i:
                all_files_dir = self.get_images_in_dir(path, [])
                if not all_files_dir:
                    continue
                for file in all_files_dir:
                    full_path = os.path.join(self.project_root, path, file)
                    image_info.append({
                        "full_path": full_path,
                        "name": file
                    })
                    continue
            else:
                break

        if not image_info:
            logger.error("No se encontraron imágenes válidas en las rutas especificadas.")
            return {}
            
        return {"image_info": image_info}
        
    def get_images_in_dir(self, input_path: str, files_list: List[str]) -> List[str]:
        files_name_dir = [file for _, _, files in os.walk(input_path) for file in files if file.endswith(self.valid_extensions)]
        if not files_name_dir:
            return []
        if not files_list:
            return files_name_dir
    
        split_names = [os.path.splitext(file) for file in files_name_dir]
        # logger.info(f"{split_names}")
        # names_to_set = [n[0] for n in split_names]
        
        # logger.info(f"NAMES TO SET: {names_to_set}")
        # logger.info(f"FILES LIST: {files_list}")
        
        files_in_dir = ["".join(name) for name in split_names if name[0] in files_list]
        # logger.info(f"INTER IDX: {files_in_dir}")
        return files_name_dir if not files_in_dir else files_in_dir