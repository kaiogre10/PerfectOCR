# core/workers/preprocessing/ink_enhancer.py
import cv2
import numpy as np
import logging
import time
from typing import Dict, Any
from core.factory.abstract_worker import ImagePrepAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.utils.image_analizer import extract_cc_metrics
from services.output_service import save_croped_image
from core.utils.image_utils import normalice_image

logger = logging.getLogger(__name__)

class InkCorrector(ImagePrepAbstractWorker):
    """Worker especializado en restaurar texto con tinta gastada o de baja intensidad."""

    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get('ink_enhancement', {})
        self.bin_interval = config["bin_interval"]
        self.kernel_threshold: int = self.worker_config.get("kernel_threshold", {})
        self.area_threshold: int = self.worker_config.get("area_threshold", {})
        self.iterations: int = self.worker_config.get("iterations", {})
        self.output = config.get("bin_full_img")
        # self.output_morph = config.get("morphology")

    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """Detecta y restaura texto con tinta gastada."""
        try:
            start_time = time.perf_counter()
            logger.debug("Mejoramiento de tinta empezado con éxito")
            image_name = manager.workflow.metadata.image_name if manager.workflow else ""

            img_obj = manager.get_full_img()
            full_img = img_obj.full_img if img_obj is not None else None
            
            if full_img is None:
                logger.error(f"No Hay full_img en el Formatter")
                return False
                
            gray_img = self._decolorate(full_img)
            kernel=np.zeros((self.kernel_threshold, self.kernel_threshold), dtype=np.uint8)
            fisrt_dil = cv2.dilate(gray_img, kernel, iterations=2) # type: ignore
            metrics, full_bin_img = extract_cc_metrics(fisrt_dil.copy(), worker_config={}, binarice=True)
            correct = self._restore_faded_ink(gray_img, full_bin_img, metrics)
            dilated = cv2.dilate(correct, None, iterations=1) # type: ignore

            if self.output:
                worker_name = context.get("worker_name") or "inker"
                output_paths = context["output_paths"]
                img_id = f"full_bin_img_{image_name}_{worker_name}"
                image_id = f"full_dilated_{image_name}_{worker_name}"
                imag_id = f"corrected_blobs_{image_name}_{worker_name}"
                id = f"decolored_{image_name}_{worker_name}"

                save_croped_image(image_name, id, gray_img, output_paths, worker_name)
                save_croped_image(image_name, id, gray_img, output_paths, worker_name)
                save_croped_image(image_name, img_id, full_bin_img, output_paths, worker_name)
                save_croped_image(image_name, image_id, fisrt_dil, output_paths, worker_name) # type: ignore
                save_croped_image(image_name, imag_id, correct, output_paths, worker_name)

            # if self.output_morph:
            #     kernel = np.ones(self.kernel_threshold, dtype=np.uint8)
            #     for i in range(0, self.iterations):
            #         opening = cv2.morphologyEx(correct.copy(), cv2.MORPH_OPEN, kernel, iterations= i+1)
            #         closing = cv2.morphologyEx(correct.copy(), cv2.MORPH_CLOSE, kernel, iterations= i+1)

            #         logger.info(f"Conteo de fondo interación: '{i+1}': Opening: '{np.count_nonzero(opening)}', Closing: '{np.count_nonzero(closing)}'")

            #         img_id = f"open_img_{image_name}_{worker_name}_{i+1}"
            #         image_id = f"close_img_{image_name}_{worker_name}_{i+1}"
            #         save_croped_image(image_name, img_id, opening, output_paths, worker_name)
            #         save_croped_image(image_name, image_id, closing, output_paths, worker_name)
                    
            logger.debug(f"Restauración de tinta completada para '{image_name}' en: {time.perf_counter() - start_time:.6f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Error en InkEnhancer: {e}", exc_info=True)
            return False

    def _restore_faded_ink(self, full_img: np.ndarray[Any, np.dtype[np.uint8]], full_bin_img: np.ndarray[Any, Any], metrics: Dict[str, Any]) -> np.ndarray[Any, Any]:
        """
        Restaura la intensidad del texto y elimina el ruido aislado.
        Para cada componente pequeño, se analiza una ventana a su alrededor. Si el borde de
        la ventana es completamente negro (fondo), se considera ruido y se elimina.
        """
        img_h, img_w = full_bin_img.shape
        cont_array_dict: Dict[int, Dict[str, Any]] = metrics.get("cont_array_dict", {})
        first_black = np.count_nonzero(full_bin_img)
        corrected_blobs = 0
        
        for pos, countours in cont_array_dict.items(): # type: ignore
            cont_area = countours["cont_area"]
                
            if self.area_threshold >= cont_area:
                cont_bbox = countours["cont_bbox"]
                cont_coords = countours["cont_coords"]
                
                x, y, w, h = cont_bbox

                # Define la ventana de análisis alrededor del blob con un padding
                win_x1 = max(0, x - self.kernel_threshold)
                win_y1 = max(0, y - self.kernel_threshold)
                win_x2 = min(img_w, x + w + self.kernel_threshold)
                win_y2 = min(img_h, y + h + self.kernel_threshold)

                # Extrae la región de la ventana
                window = full_bin_img[win_y1:win_y2, win_x1:win_x2]

                # Extrae los bordes de la ventana
                border_top = window[0, :]
                border_bottom = window[-1, :]
                border_left = window[1:-1, 0]
                border_right = window[1:-1, -1]

                # Concatena todos los píxeles del borde
                border_pixels = np.concatenate([border_top, border_bottom, border_left, border_right]).astype(np.uint8)

                # Si todos los píxeles del borde son negros (fondo), el blob está aislado
                if self.bin_interval[0] >= np.mean(border_pixels):
                # if np.all(border_pixels == 0):
                    # Pinta el blob de blanco (255) en la imagen original para corregir ruido
                    cv2.drawContours(full_img, [cont_coords], -1, color=255, thickness=cv2.FILLED) # type: ignore
                    corrected_blobs += 1
                    
                # if np.all(border_pixels == 255):
                #     cv2.drawContours(full_img, [cont_coords], -1, color=0, thickness=cv2.FILLED)
                #     corrected_blobs += 1

        logger.info(f"Total de texto: {first_black}, blobs corregidos: {corrected_blobs}")
        return full_img

    def _decolorate(self, full_img: np.ndarray[Any, Any]) -> np.ndarray[Any, np.dtype[np.uint8]]:
        """
        Elimina colores (rayones, resaltados, etc.) de la imagen, dejando solo blanco y negro.
        """
        # Detecta píxeles que no sean casi blancos ni casi negros (o sea, que tengan color) y los reemplaza por blanco.
        # Se asume imagen en BGR.
        threshold_black = 160  # Píxeles con todos los canales <= 60 se consideran negro
        threshold_white = 180 # Píxeles con todos los canales >= 200 se consideran blanco

        # Máscara para píxeles negros (todos los canales <= threshold_black)
        mask_black = np.all(full_img <= threshold_black, axis=2)
        
        # Máscara para píxeles blancos (todos los canales >= threshold_white)
        mask_white = np.all(full_img >= threshold_white, axis=2)
        
        # Máscara de píxeles válidos (negro o blanco)
        mask_valid = mask_black | mask_white
        
        # Reemplaza los píxeles de color (no válidos) por blanco
        full_img[~mask_valid] = [255, 255, 255]

        # Convierte a escala de grises para continuar el flujo normal
        gray = normalice_image(full_img.copy())
        
        if gray is not None:
            return gray
            
        else:
            logger.info("Normalice IMG devolvío imagen, Imagen en grises de cv2")
            return cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY).astype(np.uint8)