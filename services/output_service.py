# core/utils/output_service.py
import os
import json
import cv2
import logging
import numpy as np
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

def save_json(data: Dict[str, Any], output_dir: str, file_name_with_extension: str) -> Optional[str]:
    """Guarda un JSON en disco."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, file_name_with_extension)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return output_file
    except Exception as e:
        logger.error(f"Error guardando JSON: {e}")
        return None

def save_image(image: np.ndarray, output_dir: str, file_name_with_extension: str) -> Optional[str]:
    """Guarda una única imagen en disco."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        img_path = os.path.join(output_dir, file_name_with_extension)
        cv2.imwrite(img_path, image)
        return img_path
    except Exception as e:
        logger.error(f"Error guardando imagen: {e}")
        return None

def save_text(text: str, output_dir: str, file_name_with_extension: str) -> Optional[str]:
    """Guarda texto en disco."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, file_name_with_extension)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        return output_file
    except Exception as e:
        logger.error(f"Error guardando texto: {e}")
        return None