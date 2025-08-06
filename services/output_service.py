# core/utils/output_service.py
import os
import json
import cv2
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class OutputService:
    """Handler que solo ejecuta la escritura, sin lógica ni estado."""

    def save(self, data: Dict, output_dir: str, file_name_with_extension: str) -> Optional[str]:
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

    def save_images(self, images: List, output_dir: str, base_name: str) -> Optional[str]:
        """Guarda imágenes en disco."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            for i, img in enumerate(images):
                if img is not None:
                    img_filename = f"{base_name}_{i+1}.png"
                    img_path = os.path.join(output_dir, img_filename)
                    cv2.imwrite(img_path, img)
            return output_dir
        except Exception as e:
            logger.error(f"Error guardando imágenes: {e}")
            return None

    def save_text(self, text: str, output_dir: str, file_name_with_extension: str) -> Optional[str]:
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