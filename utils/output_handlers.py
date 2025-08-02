# core/utils/output_handlers.py
import os
import json
import cv2
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class OutputPathContainer:
    """Contenedor exclusivo para la ruta de output"""
    output_path: str = ""
    is_active: bool = False

# Instancia global que solo ConfigManager puede activar
output_path_container: Optional[OutputPathContainer] = None

def dump_json(data: Dict, base_name: str, output_type: str) -> Optional[str]:
    """Vomita JSON al disco. Solo ejecuta, no piensa."""
    if not output_path_container or not output_path_container.is_active:
        return None
    
    try:
        output_file = f"{output_path_container.output_path}/{base_name}_{output_type}.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        return output_file
    except Exception as e:
        logger.error(f"Error vomitando JSON: {e}")
        return None

def dump_images(images: List, base_name: str, output_type: str) -> Optional[str]:
    """Vomita imágenes al disco. Solo ejecuta, no piensa."""
    if not output_path_container or not output_path_container.is_active:
        return None
    
    try:
        output_folder = f"{output_path_container.output_path}/{base_name}_{output_type}"
        os.makedirs(output_folder, exist_ok=True)
        
        for i, img in enumerate(images):
            if img is not None:
                img_filename = f"{base_name}_{output_type}_{i+1}.png"
                img_path = os.path.join(output_folder, img_filename)
                cv2.imwrite(img_path, img)
        
        return output_folder
    except Exception as e:
        logger.error(f"Error vomitando imágenes: {e}")
        return None

def dump_text(text: str, base_name: str, output_type: str) -> Optional[str]:
    """Vomita texto al disco. Solo ejecuta, no piensa."""
    if not output_path_container or not output_path_container.is_active:
        return None
    
    try:
        output_file = f"{output_path_container.output_path}/{base_name}_{output_type}.txt"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        return output_file
    except Exception as e:
        logger.error(f"Error vomitando texto: {e}")
        return None