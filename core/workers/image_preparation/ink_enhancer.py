# PerfectOCR/core/workers/preprocessing/ink_enhancer.py
import cv2
import numpy as np
import logging
import time
from typing import Dict, Any, Tuple
from core.factory.abstract_worker import ImagePrepAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.utils.image_analizer import extract_cc_metrics

logger = logging.getLogger(__name__)

class InkEnhancer(ImagePrepAbstractWorker):
    """Worker especializado en restaurar texto con tinta gastada o de baja intensidad."""

    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get('ink_enhancement', {})
        self.bin_interval = config["bin_interval"]
        self.kernel_threshold: int = self.worker_config.get("kernel_threshold", 3)
self.kernel = np.ones((self.kernel_threshold, self.kernel_threshold), np.uint8) 
        self.area_threshold: int = self.worker_config.get("area_threshold", 12)
        self.iterations: int = self.worker_config.get("iterations", 2)
        self.output = config.get("bin_full_img", False)

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
                    
            _, full_bin_img = extract_cc_metrics(full_img, worker_config={}, binarice=True)
            # if full_bin_img is None:
            #     logger.info("Nose devolvio imagen binarizada")
            #     return False
            
            if self.output:
                from services.output_service import save_croped_image
                img_id = f"full_bin_img_{image_name}_{worker_name}"
                save_croped_image(image_name, img_id, full_bin_img, output_paths, worker_name, method="binarized")

            for i in range(0, self.iterations):
                opening = cv2.morphologyEx(full_bin_img.copy(), cv2.MORPH_OPEN, self.kernel, iterations= i+1)
                closing = cv2.morphologyEx(full_bin_img.copy(), cv2.MORPH_CLOSE, self.kernel, iterations= i+1)

                logger.info(f"Conteo de fondo interación: '{i+1}': Opening: '{np.count_nonzero(opening)}', Closing: '{np.count_nonzero(closing)}'")

                if self.output:
                    from services.output_service import save_croped_image
                    img_id = f"open_img_{image_name}_{worker_name}_{i+1}"
                    image_id = f"close_img_{image_name}_{worker_name}_{i+1}"
                    save_croped_image(image_name, img_id, opening, output_paths, worker_name, method="opening")
                    save_croped_image(image_name, image_id, closing, output_paths, worker_name, method="closing")

                # metrics, _ = extract_cc_metrics(full_bin_img_rest, worker_config={}, binarice=False)
                # if full_bin_img is None:
                #     return False
                    
            logger.debug(f"Restauración de tinta completada para '{image_name}' en: {time.perf_counter() - start_time:.6f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Error en InkEnhancer: {e}", exc_info=True)
            return False

    def _restore_faded_ink(self, full_bin_img: np.ndarray[Any, Any], metrics: Dict[str, Any]) -> np.ndarray[Any, Any]:
        """Restaura la intensidad del texto con tinta gastada."""
        
        cont_array_dict: Dict[int, Dict[str, Any]] = metrics["cont_array_dict"]
        
        for pos, countours in cont_array_dict.items():
            cont_coords = countours["cont_coords"]
            # convex_hull = countours["convex_hull"]
            # hull_area = countours["hull_area"]
            cont_area = countours["cont_area"]
            cont_bbox = countours["cont_bbox"]
                
            if self.area_threshold >= cont_area:
                cont_bbox = countours["cont_bbox"]
                # Crear una copia de la imagen para dibujar
                img_with_rect = bin_img.copy()
                
                # Asumir que cont_bbox es [x, y, width, height]
                if len(cont_bbox) == 4:
                    x, y, w, h = cont_bbox
                    cx = int((x + w) / 2)
                    cy = int((y + h) / 2)

                    # Crear ventana fija usando array
                    window_size = self.kernel_threshold
                    half_size = window_size // 2
                    
                    # Calcular límites de la ventana
                    start_y = max(0, cy - half_size)
                    end_y = min(img_h, cy + half_size)
                    start_x = max(0, cx - half_size)
                    end_x = min(img_w, cx + half_size)
                    
                    # Crear ventana fija (rellenar con ceros si está en el borde)
                    window = np.zeros((window_size, window_size), dtype=np.uint8)
                    
                    # Extraer la región disponible
                    # region = bin_img[start_y:end_y, start_x:end_x]
                    region = cropped_img[start_y:end_y, start_x:end_x]
                    
                    # Calcular offsets para centrar en la ventana fija
                    offset_y = half_size - (cy - start_y)
                    offset_x = half_size - (cx - start_x)
                    
                    # Colocar la región en la ventana fija
                    window[offset_y:offset_y + region.shape[0], 
                           offset_x:offset_x + region.shape[1]] = region
                    
                    # Dibujar rectángulo en la imagen
                    cv2.rectangle(img_with_rect, (start_x, start_y), (end_x, end_y), (255), 1)
                    
                else:
                    logger.warning(f"cont_bbox formato inesperado: {cont_bbox}")
                    continue
                
                logger.info(f"VENTANA{window.shape}")
                mean = cv2.mean(window)[0]
                white_pixels = np.sum(window >= self.bin_interval[1])
                black_pixels = np.sum(self.bin_interval[0] >= window)
                if not white_pixels and not black_pixels:
                    grey_pixels = window.size - white_pixels - black_pixels
                    logger.info(f"Grises: {grey_pixels}")

                logger.info(f"MEAN: {mean:.4f}, Blancos: {white_pixels}, Negros: {black_pixels}")

            convex_array = np.array(convex_hull).reshape(-1, 2)

            convex_array_reshape = np.delete(convex_array, 0, axis=0).astype(np.int32)
            convex_list.append({
                "convex_array_reshape":