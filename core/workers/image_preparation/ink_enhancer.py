# core/workers/preprocessing/ink_enhancer.py
import cv2
import numpy as np
import logging
import time
from typing import Dict, Any, List
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
        # self.kernel = config["morph_kernel"]
        self.isolation_range = self.worker_config["isolation_range"]
        self.iterations: int = self.worker_config.get("iterations", {})
        self.window_size = self.worker_config["window_size"]
        self.threshold_black: int = self.worker_config.get("threshold_black", {})
        self.threshold_white: int = self.worker_config.get("threshold_white", {})
        self.output = config.get("bin_full_img")

    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """Detecta y restaura texto con tinta gastada."""
        try:
            start_time = time.perf_counter()
            logger.debug("Mejoramiento de tinta empezado con éxito")
            image_name = manager.workflow.metadata.image_name if manager.workflow else ""
            context["image_name"]= image_name

            img_obj = manager.get_full_img()
            full_img = img_obj.full_img if img_obj is not None else None
            
            if full_img is None:
                logger.error(f"No Hay full_img en el Formatter")
                return False
                
            gray_img = self._decolorate(full_img)
            metrics = extract_cc_metrics(gray_img.copy(), binarice=False)
            correct, contours_list, blacked_contours = self.alternate_restore(gray_img.copy(), metrics)
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.kernel)

            # dilated = cv2.morphologyEx(correct.copy(), cv2.MORPH_OPEN, kernel, iterations=2)
            
            if not manager.update_full_img(True, correct):
                logger.warning("No se actualizo imagen en escala de grises del enhancer", exc_info=True)
                return False
                
            else:

                if self.output:
                    worker_name = context.get("worker_name") or "inker"
                    output_paths = context["output_paths"]
                    imag_id = f"corrected_blobs_{image_name}_{worker_name}"
                    image_id = f"contours_{image_name}_{worker_name}"
                    # image_id = f"dilated_{image_name}_{worker_name}"
                    # imge_id = f"eroded_{image_name}_{worker_name}"

                    # save_croped_image(image_name, id, gray_img, output_paths, worker_name)
                    # save_croped_image(image_name, image_id, dilated, output_paths, worker_name)
                    save_croped_image(image_name, imag_id, correct, output_paths, worker_name)
                    save_shapes(image_name, image_id, gray_img, output_paths, contours_list, contours2=blacked_contours)
                        
                logger.debug(f"Restauración de tinta completada para '{image_name}' en: {time.perf_counter() - start_time:.6f}s")
                
                return True
            
        except Exception as e:
            logger.error(f"Error en InkEnhancer: {e}", exc_info=True)
            return False
        
    def alternate_restore(self, gray_img: np.ndarray[Any, Any], metrics: Dict[str, Any]):
        """
        Alterna entre restaurar la imagen original y la invertida, reutilizando la lógica de _restore_faded_ink.
        Devuelve: imagen corregida, contours_list, cleaned_blobs, blacked_contours.
        """
        contours_list: List[np.ndarray[Any, Any]] = []
        blacked_contours: List[np.ndarray[Any, Any]] = []
        cleaned_blobs: Dict[int, Dict[str, np.ndarray[Any, np.dtype[np.int32]]]] = {}
        
        for i in range(self.iterations):
            if i % 2 == 0:
                logger.info("Imagen normal")
                window_size = self.window_size[1] + i
                gray_img, c_list, c_blobs = self._restore_faded_ink(gray_img, metrics, window_size, self.isolation_range[1])
                logger.debug(f"Tinta eliminada: {len(c_list)} blobs")
                contours_list.extend(c_list)
            else:
                gray_img_inv = gray_img - 255
                logger.info("Inversión")
                window_size = self.window_size[0] + i
                gray_img, c_list, c_blobs = self._restore_faded_ink(gray_img_inv, metrics, window_size, self.isolation_range[0])
                logger.debug(f"Tinta restaurada: {len(c_list)} blobs")
                blacked_contours.extend(c_list)
                gray_img = 255 - gray_img_inv
            
            cleaned_blobs.update(c_blobs)

        return gray_img, contours_list, blacked_contours

    def _restore_faded_ink(self, gray_img: np.ndarray[Any, Any], metrics: Dict[str, Any], window_size: int, isolation_range: int):
        """
        Restaura la intensidad del texto y elimina el ruido aislado usando Inpainting.
        """
        img_h, img_w = gray_img.shape
        cont_array_dict: Dict[int, Dict[str, Any]] = metrics.get("cont_array_dict", {})
        bin_edges = metrics["bin_edges"]

        k_size = 2 * window_size + 1
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        
        # logger.info(f"Umbral dinámico: {bin_edges[1]}")

        corrected_blobs = 0
        
        contours_list: List[np.ndarray[Any, Any]] = []
        cleaned_blobs: Dict[int, Dict[str, np.ndarray[Any, np.dtype[np.int32]]]] = {}
        
        # 1. PRE-FILTRADO: Identificamos solo los candidatos a ruido (blobs pequeños)
        noise_candidates = [
            pos for pos, data in cont_array_dict.items() 
            if data["cont_area"] < bin_edges[1]
        ]
        
        bboxes: List[float] = []
        for i in range(0, self.iterations): #type: ignore
            blobs_removed_this_pass = 0
            next_pass_candidates: List[int] = []

            for pos in noise_candidates:
                contours = cont_array_dict[pos]
                cont_bbox = contours["cont_bbox"]
                cont_coords = contours["cont_coords"]
                cont_area = contours["cont_area"]
                blob_centroid = contours["blob_centroid"]
                
                x, y, w, h = cont_bbox

                # [Lógica de Ventana] 
                # Siempre tomamos una vista fresca de gray_img para ver los cambios recientes
                win_x1 = max(0, x - window_size)
                win_y1 = max(0, y - window_size)
                win_x2 = min(img_w, x + w + window_size)
                win_y2 = min(img_h, y + h + window_size)

                window_gray = gray_img[win_y1:win_y2, win_x1:win_x2]

                # Ajuste de coordenadas al sistema local
                cont_coords_window = cont_coords - np.array([[win_x1, win_y1]])

                # 1. Máscara del blob
                mask_blob = np.zeros(window_gray.shape, dtype=np.uint8)
                cv2.drawContours(mask_blob, [cont_coords_window], -1, 255, thickness=cv2.FILLED) #type: ignore

                # 2. Anillo exterior
                mask_dilated = cv2.dilate(mask_blob, kernel, iterations=1)
                mask_surroundings = cv2.subtract(mask_dilated, mask_blob)
                
                # 3. Análisis de píxeles circundantes
                surrounding_pixels = window_gray[mask_surroundings == 255]

                window_mean = np.mean(surrounding_pixels)

                # logger.info(f"Media PÍXELES: {window_mean}")

                # Si el promedio es alto (blanco), es ruido aislado
                if window_mean > isolation_range:
                    array_coords = np.array(cont_coords, np.int32)
                    contours_list.append(array_coords)
                    
                    # ACCIÓN: Pintamos BLANCO sobre la imagen original (elimina ruido)
                    cv2.drawContours(gray_img, [cont_coords], -1, color=255, thickness=cv2.FILLED)#type: ignore
                    corrected_blobs += 1
                    blobs_removed_this_pass += 1

                    bboxes.append(cont_bbox)
                    cleaned_blobs[pos] = {
                        "cont_coords": cont_coords,
                        "cont_bbox": cont_bbox,
                        "cont_area": cont_area,
                        "blob_centroid": blob_centroid,
                        "surrounding_pixels": surrounding_pixels
                    }
                    
                else:
                    # Si todavía no parece ruido ni tinta débil, lo guardamos para intentarlo en la siguiente pasada.
                    next_pass_candidates.append(pos)

            # Actualizamos la lista de candidatos para la siguiente vuelta (se va reduciendo)
            noise_candidates = next_pass_candidates

            # Si en esta vuelta no pudimos limpiar nada, terminamos para no perder tiempo
            if blobs_removed_this_pass == 0:
                break

        return gray_img, contours_list, cleaned_blobs

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