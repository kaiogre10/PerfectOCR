# PerfectOCR/core/workers/preprocessing/ink_enhancer.py
import cv2
import numpy as np
import logging
import time
from typing import Dict, Any
from core.factory.abstract_worker import ImagePrepAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.utils.image_analizer import extract_cc_metrics
from core.utils.image_utils import binarice_img
from services.output_service import save_croped_image

logger = logging.getLogger(__name__)

class InkEnhancer(ImagePrepAbstractWorker):
    """Worker especializado en restaurar texto con tinta gastada o de baja intensidad."""

    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get('ink_enhancement', {})
        self.bin_interval = config["bin_interval"]
        self.kernel_threshold: int = self.worker_config.get("kernel_threshold", {})
        self.kernel = np.ones((self.kernel_threshold, self.kernel_threshold), np.uint8) 
        self.area_threshold: int = self.worker_config.get("area_threshold", {})
        self.iterations: int = self.worker_config.get("iterations", {})
        self.output = config.get("bin_full_img")
        self.output_morph = config.get("morphology")

    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """Detecta y restaura texto con tinta gastada."""
        try:
            start_time = time.perf_counter()
            logger.info("Mejoramiento de tinta empezado con éxito")
            image_name = manager.workflow.metadata.image_name if manager.workflow else ""
            worker_name = context.get("worker_name") or "inker"
            output_paths = context["output_paths"]

            img_obj = manager.get_full_img()
            full_img = img_obj.full_img if img_obj is not None else None
            if full_img is None:
                logger.error(f"No Hay full_img en el Formatter")
                return False
                   
          #  eroded = cv2.erode(full_img.copy(), None, iterations=1)
            #bin_eroded = binarice_img(eroded, worker_config={})

            metrics, full_bin_img = extract_cc_metrics(full_img, worker_config={}, binarice=True)
            corrected = self._restore_faded_ink(full_bin_img.copy(), metrics)
            bin_eroded = cv2.dilate(corrected.copy(), None, iterations=1)

            if not manager.update_full_img(corrected=True, full_img=bin_eroded):
                logger.info(f"Imagen no guardada")


            logger.info(f"Imagen corregida guardada")
            if self.output:
                img_id = f"full_bin_img_{image_name}_{worker_name}"
                image_id = f"full_eroded_{image_name}_{worker_name}"
                save_croped_image(image_name, img_id, full_bin_img, output_paths, worker_name, method="binarized")
                save_croped_image(image_name, image_id, bin_eroded, output_paths, worker_name, method="eroded")
               
                img_id = f"corrected_blobs_{image_name}_{worker_name}"
                save_croped_image(image_name, img_id, corrected, output_paths, worker_name, method="blobs")

            if self.output_morph:
                for i in range(0, self.iterations):
                    opening = cv2.morphologyEx(corrected.copy(), cv2.MORPH_OPEN, self.kernel, iterations= i+1)
                    closing = cv2.morphologyEx(corrected.copy(), cv2.MORPH_CLOSE, self.kernel, iterations= i+1)

                    logger.info(f"Conteo de fondo interación: '{i+1}': Opening: '{np.count_nonzero(opening)}', Closing: '{np.count_nonzero(closing)}'")

                    img_id = f"open_img_{image_name}_{worker_name}_{i+1}"
                    image_id = f"close_img_{image_name}_{worker_name}_{i+1}"
                    save_croped_image(image_name, img_id, opening, output_paths, worker_name, method="opening")
                    save_croped_image(image_name, image_id, closing, output_paths, worker_name, method="closing")
                    
            logger.debug(f"Restauración de tinta completada para '{image_name}' en: {time.perf_counter() - start_time:.6f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Error en InkEnhancer: {e}", exc_info=True)
            return False

    def _restore_faded_ink(self, full_bin_img: np.ndarray[Any, Any], metrics: Dict[str, Any]) -> np.ndarray[Any, Any]:
        """
        Restaura la intensidad del texto y elimina el ruido aislado.
        Para cada componente pequeño, se analiza una ventana a su alrededor. Si el borde de
        la ventana es completamente negro (fondo), se considera ruido y se elimina.
        """
        img_h, img_w = full_bin_img.shape
        cont_array_dict: Dict[int, Dict[str, Any]] = metrics["cont_array_dict"]
        first_black = np.count_nonzero(full_bin_img)
        
        for pos, countours in cont_array_dict.items():
            cont_area = countours["cont_area"]
            #convex_hull = countours["convex_hull"]
            cont_coords = countours["cont_coords"]
                
            if self.area_threshold >= cont_area:
                cont_bbox = countours["cont_bbox"]
                
                x, y, w, h = cont_bbox

                # Define la ventana de análisis alrededor del blob con un padding
                win_x1 = max(0, x - self.kernel_threshold)
                win_y1 = max(0, y - self.kernel_threshold)
                win_x2 = min(img_w, x + w + self.kernel_threshold )
                win_y2 = min(img_h, y + h + self.kernel_threshold)

                # Extrae la región de la ventana
                window = full_bin_img[win_y1:win_y2, win_x1:win_x2]

                # Extrae los bordes de la ventana
                border_top = window[0, :]
                border_bottom = window[-1, :]
                border_left = window[1:-1, 0]
                border_right = window[1:-1, -1]

                # Concatena todos los píxeles del borde
                border_pixels = np.concatenate([border_top, border_bottom, border_left, border_right])
                # Si la suma de los píxeles del borde es 0, significa que todos son negros (fondo) y el blob está aislado.
                if np.all(border_pixels) == 0:
                    # Rellena el contorno del blob con negro (0) en la imagen original
                    cv2.drawContours(full_bin_img, [cont_coords], -1, color=0, thickness=cv2.FILLED)
 
        logger.info(f"Total de fondo incial: {first_black}, corregido: {first_black - np.count_nonzero(full_bin_img)}")
        return full_bin_img

