# PerfectOCR/core/workflow/preprocessing/poly_gone.py
import cv2
import numpy as np
import logging
import os
from typing import Tuple, List, Dict, Any
from core.workspace.utils.output_handlers import ImageOutputHandler

logger = logging.getLogger(__name__)

class PolygonExtractor:
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.config = config
        self.image_saver = ImageOutputHandler()

    def _add_cropped_images_to_lines(self, deskewed_img: np.ndarray, reconstructed_lines: List[Dict], metadata: Dict, input_filename: str = "", output_config: Dict = None) -> Dict:
        """
        Añade las imágenes recortadas de cada línea al diccionario correspondiente y opcionalmente las guarda.
        """
        if not reconstructed_lines:
            return {"lines": [], "metadata": metadata}

        img_h, img_w = deskewed_img.shape[:2]
        padding = self.config.get('cropping_padding', 5)
        
        logger.info(f"Recortando {len(reconstructed_lines)} líneas de imagen de {img_w}x{img_h}")

        # Verificar si el guardado de imágenes está habilitado
        should_save_images = False
        if output_config:
            enabled_outputs = output_config.get('enabled_outputs', {})
            should_save_images = enabled_outputs.get('cropped_line_images', False)

        output_folder = None
        base_name = "imagen"
        
        if should_save_images:
            # Crear directorio para las imágenes recortadas
            output_folder = os.path.join(self.project_root, "output", "cropped_lines")
            os.makedirs(output_folder, exist_ok=True)
            
            # Obtener nombre base del archivo
            if input_filename:
                base_name = os.path.splitext(os.path.basename(input_filename))[0]

        for line_idx, line in enumerate(reconstructed_lines):
            try:
                # Obtener el bounding box de la línea
                min_x, min_y, max_x, max_y = map(int, line["line_bbox"])
                
                # Aplicar padding y asegurar límites de la imagen
                x1 = max(0, min_x - padding)
                y1 = max(0, min_y - padding)
                x2 = min(img_w, max_x + padding)
                y2 = min(img_h, max_y + padding)
                
                # Verificar que las coordenadas sean válidas
                if x2 > x1 and y2 > y1:
                    # Recortar la imagen de la línea
                    cropped_img = deskewed_img[y1:y2, x1:x2]
                    
                    if cropped_img.size > 0:
                        line["cropped_img"] = cropped_img
                        
                        # Guardar la imagen recortada solo si está habilitado
                        if should_save_images:
                            line_number = line_idx + 1  # Empezar desde 1
                            output_filename = f"{base_name}_linea_{line_number:03d}.png"
                            saved_path = self.image_saver.save(cropped_img, output_folder, output_filename)
                            
                            if saved_path:
                                logger.info(f"Línea {line_number} guardada: {saved_path}")
                                line["saved_image_path"] = saved_path
                        
                        logger.debug(f"Línea {line['line_id']} recortada: {cropped_img.shape}")
                    else:
                        logger.warning(f"Imagen vacía para línea {line['line_id']}")
                        line["cropped_img"] = None
                else:
                    logger.warning(f"Coordenadas inválidas para línea {line['line_id']}: bbox={line['line_bbox']}")
                    line["cropped_img"] = None
                    
            except Exception as e:
                logger.error(f"Error recortando línea {line.get('line_id', 'desconocida')}: {e}")
                line["cropped_img"] = None

        if should_save_images:
            logger.info(f"Guardado de imágenes recortadas habilitado: {len(reconstructed_lines)} imágenes procesadas")
        else:
            logger.debug("Guardado de imágenes recortadas deshabilitado")

        return {
            "lines": reconstructed_lines,
            "metadata": metadata
        }