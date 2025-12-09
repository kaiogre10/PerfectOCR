# core/workers/preprocessing/ink_enhancer.py
import cv2
import numpy as np
import logging
import time
from typing import Dict, Any, List, Tuple
from core.factory.abstract_worker import ImagePrepAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.utils.image_analizer import extract_cc_metrics
from services.output_service import save_croped_image, save_shapes
from core.utils.image_utils import normalice_image

logger = logging.getLogger(__name__)

class InkCorrector(ImagePrepAbstractWorker):
    """Worker especializado en restaurar texto con tinta gastada o de baja intensidad."""

    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get('ink_enhancement', {})
        self.isolation_range = self.worker_config["isolation_range"]
        self.iterations: int = self.worker_config.get("iterations", {})
        self.window_size: int = self.worker_config.get("window_size", {})
        self.threshold_black: int = self.worker_config.get("threshold_black", {})
        self.threshold_white: int = self.worker_config.get("threshold_white", {})
        self.output = config.get("bin_full_img")

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
            metrics = extract_cc_metrics(gray_img.copy(), worker_config={}, binarice=False)
            correct, contours_list = self._restore_faded_ink(gray_img.copy(), metrics[0])
            
            if not manager.update_full_img(True, correct):
                logger.warning("No se actualizo imagen en escala de grises del enhancer", exc_info=True)
                return False
                
            else:
                # dilated = cv2.dilate(full_bin_img, kernel, iterations=self.iterations, borderType=cv2.BORDER_CONSTANT, borderValue=[0, 0, 255])

                if self.output:
                    worker_name = context.get("worker_name") or "inker"
                    output_paths = context["output_paths"]
                    # imag_id = f"corrected_blobs_{image_name}_{worker_name}"
                    image_id = f"contours_{image_name}_{worker_name}"
                    # image_id = f"dilated_{image_name}_{worker_name}"
                    # imge_id = f"eroded_{image_name}_{worker_name}"

                    # save_croped_image(image_name, id, gray_img, output_paths, worker_name)
                    # save_croped_image(image_name, image_id, dilated, output_paths, worker_name)
                    # save_croped_image(image_name, imag_id, correct, output_paths, worker_name)
                    save_shapes(image_name, image_id, gray_img, output_paths, worker_name, contours_list, contours2=None)
                        
                logger.debug(f"Restauración de tinta completada para '{image_name}' en: {time.perf_counter() - start_time:.6f}s")
                
                return True
            
        except Exception as e:
            logger.error(f"Error en InkEnhancer: {e}", exc_info=True)
            return False

    def _restore_faded_ink(self, gray_img: np.ndarray[Any, Any], metrics: Dict[str, Any]) -> Tuple[np.ndarray[Any, Any], List[np.ndarray[Any, Any]]]:
        """
        Restaura la intensidad del texto y elimina el ruido aislado usando Inpainting.
        """
        img_h, img_w = gray_img.shape
        cont_array_dict: Dict[int, Dict[str, Any]] = metrics.get("cont_array_dict", {})
        bin_edges = metrics["bin_edges"]
        k_size = 2 * self.window_size + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        
        logger.info(f"Umbral dinámico: {bin_edges[1]}")

        corrected_blobs = 0
        
        contours_list: List[np.ndarray[Any, Any]] = []
        cleaned_blobs: Dict[int, Dict[str, np.ndarray[Any, np.dtype[np.int32]]] | float]= {}
        for pos, countours in cont_array_dict.items():
            cont_area = countours["cont_area"]
            
            # Su lógica de filtro: Si el área cae en el primer bin de ruido
            if bin_edges[1] > cont_area:
                cont_bbox = countours["cont_bbox"]
                cont_coords = countours["cont_coords"]
                
                x, y, w, h = cont_bbox

                # [Lógica de Ventana de Aislamiento]
                win_x1 = max(0, x - self.window_size)
                win_y1 = max(0, y - self.window_size)
                win_x2 = min(img_w, x + w + self.window_size)
                win_y2 = min(img_h, y + h + self.window_size)

                window_gray = gray_img[win_y1:win_y2, win_x1:win_x2]

                # Ajustamos las coordenadas del contorno al sistema local de la ventana
                cont_coords_window = cont_coords - np.array([[win_x1, win_y1]])

                # 1. Crear máscara del blob exacto
                mask_blob = np.zeros(window_gray.shape, dtype=np.uint8)
                cv2.drawContours(mask_blob, [cont_coords_window], -1, 255, thickness=cv2.FILLED)

                # 2. Dilatar para obtener el área circundante exacta
                # Usamos iterations=1 porque el kernel ya tiene el tamaño correcto
                mask_dilated = cv2.dilate(mask_blob, kernel, iterations=self.iterations)

                # 3. Obtener solo el anillo exterior (Dilatado - Original)
                mask_surroundings = cv2.subtract(mask_dilated, mask_blob)
                
                # 4. Extraer los valores de los píxeles que rodean la forma
                surrounding_pixels = window_gray[mask_surroundings == 255]

                if np.mean(surrounding_pixels) >= self.isolation_range[1]:
                    array_coords = np.array(cont_coords, np.int32)
                    contours_list.append(array_coords)
                    
                    # Pintamos sobre la imagen original (gray_img o full_bin_img según necesites)
                    cv2.drawContours(gray_img, [cont_coords], -1, color=255, thickness=cv2.FILLED)
                    corrected_blobs += 1
                    
                cleaned_blobs[pos] = {
                    "cont_coords": cont_coords,
                    "cont_bbox": cont_bbox,
                    "cont_area": cont_area,
                }

        logger.info(f"Blobs corregidos: {corrected_blobs}")
        
        return gray_img, contours_list

    def _decolorate(self, full_img: np.ndarray[Any, Any]) -> np.ndarray[Any, np.dtype[np.uint8]]:
        """
        Elimina colores (rayones, resaltados, etc.) de la imagen, dejando solo blanco y negro.
        """
        # Máscara para píxeles negros (todos los canales <= threshold_black)
        mask_black = np.all(full_img <= self.threshold_black, axis=2)
        
        # Máscara para píxeles blancos (todos los canales >= threshold_white)
        mask_white = np.all(full_img >= self.threshold_white, axis=2)
        
        # Máscara de píxeles válidos (negro o blanco)
        mask_valid = mask_black | mask_white
        
        # Reemplaza los píxeles de color (no válidos) por blanco
        full_img[~mask_valid] = [255, 255, 255]

        # Convierte a escala de grises para continuar el flujo normal
        gray = normalice_image(full_img.copy())
        
        if gray is not None:
            return gray
            
        else:
            logger.warning("Normalice IMG devolvío imagen, Imagen en grises de cv2")
            return cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY).astype(np.uint8)