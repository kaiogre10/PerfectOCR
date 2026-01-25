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
        worker_config = config.get('ink_enhancement', {})
        self.noise_kernel= np.array(worker_config["noise_kernel"])
        self.retorer_kernel = np.array(worker_config["retorer_kernel"])
        self.isolation_range = worker_config["isolation_range"]
        self.start_restoring = worker_config.get("start_restoring")
        self.iterations: int = worker_config.get("iterations")
        self.threshold_black: int = worker_config.get("threshold_black")
        self.threshold_white: int = worker_config.get("threshold_white")
        self.output = config.get("bin_full_img")

    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """Detecta y restaura texto con tinta gastada."""
        try:
            start_time = time.perf_counter()
            logger.debug("Mejoramiento de tinta empezado con éxito")
            
            if not self.noise_kernel[0] == self.noise_kernel[1] or self.retorer_kernel[0] == self.retorer_kernel[1]:
                logger.warning(f"Kernels mal configurados")
                return False
            
            image_name = manager.workflow.metadata.image_name if manager.workflow else ""
            context["image_name"]= image_name

            img_obj = manager.get_full_img()
            full_img = img_obj.full_img if img_obj is not None else None
            
            if full_img is None:
                logger.error(f"No Hay full_img en el Formatter")
                return False
                
            grey_img = self._decolorate(full_img)
            metrics = extract_cc_metrics(grey_img, binarice=False)
            lines_cont, gray_img = self.compare_areas(metrics, grey_img)
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.retorer_kernel[0], self.retorer_kernel[1]))
            # dilated = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel, iterations=1)

            correct, contours_list, blacked_contours = self.alternate_restore(gray_img)
            
            if not manager.update_full_img(True, correct):
                logger.warning("No se actualizo imagen en escala de grises del enhancer", exc_info=True)
                return False    
            else:
                if self.output:
                    worker_name = context.get("worker_name") or "inker"
                    output_paths = context["output_paths"]
                    imag_id = f"corrected_blobs_{image_name}_{worker_name}"
                    image_id = f"contours_{image_name}_{worker_name}"
                    # image_idd = f"dilated_{image_name}_{worker_name}"
                    # imge_id = f"eroded_{image_name}_{worker_name}"
                    line_cont_id= f"lines_{image_name}_{worker_name}"

                    # save_croped_image(image_name, id, gray_img, output_paths, worker_name)
                    # save_croped_image(image_name, image_idd, dilated, output_paths, worker_name)
                    save_croped_image(image_name, imag_id, correct, output_paths, worker_name)
                    save_shapes(image_name, image_id, grey_img, output_paths, contours_list, contours2=blacked_contours)
                    save_shapes(image_name, line_cont_id, gray_img, output_paths, lines_cont, contours2=[])
                        
                logger.debug(f"Restauración de tinta completada para '{image_name}' en: {time.perf_counter() - start_time:.6f}s")
                
                return True
            
        except Exception as e:
            logger.error(f"Error en InkEnhancer: {e}", exc_info=True)
            return False
        
    def alternate_restore(self, gray_img: np.ndarray[Any, Any]):
        """
        Por cada iteración ejecuta 2 fases (clean/restore) en el orden dictado por start_restoring.
        metrics: se calcula UNA sola vez por iteración.
        """
        img_dims = gray_img.shape
        contours_list: List[np.ndarray[Any, Any]] = []
        blacked_contours: List[np.ndarray[Any, Any]] = []
        cleaned_blobs: Dict[int, Dict[str, np.ndarray[Any, np.dtype[np.int32]]]] = {}

        for i in range(self.iterations):
            metrics = extract_cc_metrics(gray_img, binarice=False)

            phases = ("restore", "clean") if self.start_restoring else ("clean", "restore")

            for phase in phases:
                if phase == "clean":
                    # logger.info("Limpiando imagen")
                    kernel = 2 * (self.retorer_kernel + i) + 1
                    gray_img, c_list, c_blobs = self._restore_faded_ink(gray_img, self.isolation_range[1], kernel, cv2.MORPH_RECT, img_dims, metrics)
                    contours_list.extend(c_list)
                    logger.info(f"Cantidad de correcciones: {len(c_blobs)}")
                else:
                    # logger.info("Restaurando tinta")
                    kernel = 2 * (self.noise_kernel + i) + 1
                    gray_img_inv = 255 - gray_img

                    # FIX: conservar la imagen procesada (inv)
                    gray_img_inv, c_list, c_blobs = self._restore_faded_ink(gray_img_inv, self.isolation_range[0], kernel, cv2.MORPH_ELLIPSE, img_dims, metrics)
                    blacked_contours.extend(c_list)
                    logger.info(f"Cantidad de mejoras: {len(c_blobs)}")

                    # FIX: volver al dominio normal usando la invertida YA procesada
                    gray_img = 255 - gray_img_inv

                cleaned_blobs.update(c_blobs)

        return gray_img, contours_list, blacked_contours

    def _restore_faded_ink(self, gray_img: np.ndarray[Any, Any], isolation_range: int, kernel_shape: np.ndarray[Any, Any], k_shape: int, img_dims: Any, metrics: Dict[str, Any]):
        """
        Restaura la intensidad del texto y elimina el ruido aislado usando Inpainting.
        """
        img_h, img_w = img_dims
        cont_array_dict: Dict[int, Dict[str, Any]] = metrics.get("cont_array_dict", {})
        bin_edges = metrics["bin_edges"]
        
        kernel = cv2.getStructuringElement(k_shape, (kernel_shape[0], kernel_shape[1]))
        
        # logger.info(f"Umbral dinámico: {bin_edges[1]}")

        corrected_blobs = 0
        
        contours_list: List[np.ndarray[Any, Any]] = []
        cleaned_blobs: Dict[int, Dict[str, np.ndarray[Any, np.dtype[np.int32]]]] = {}
        
        # 1. PRE-FILTRADO: Identificamos solo los candidatos a ruido (blobs pequeños)
        noise_candidates = [
            pos for pos, data in cont_array_dict.items() 
            if data["cont_area"] < bin_edges[1]
        ]

        for pos in noise_candidates:
            contours = cont_array_dict[pos]
            cont_bbox = contours["cont_bbox"]
            cont_coords = contours["cont_coords"]
            cont_area = contours["cont_area"]
            blob_centroid = contours["blob_centroid"]
            
            x, y, w, h = cont_bbox

            # [Lógica de Ventana] 
            # Siempre tomamos una vista fresca de gray_img para ver los cambios recientes
            win_x1 = max(0, x - kernel_shape[1])
            win_y1 = max(0, y - kernel_shape[0])
            win_x2 = min(img_w, x + w + kernel_shape[1])
            win_y2 = min(img_h, y + h + kernel_shape[0])

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

            if surrounding_pixels.size == 0:
                continue

            window_mean = np.mean(surrounding_pixels)

            # logger.info(f"Media PÍXELES: {window_mean}")

            # Si el promedio es alto (blanco), es ruido aislado
            if window_mean > isolation_range:
                array_coords = np.array(cont_coords, np.int32)
                contours_list.append(array_coords)
                
                # ACCIÓN: Pintamos BLANCO sobre la imagen original (elimina ruido)
                cv2.drawContours(gray_img, [cont_coords], -1, color=255, thickness=cv2.FILLED)#type: ignore
                corrected_blobs += 1

                cleaned_blobs[pos] = {
                    "cont_coords": cont_coords,
                    "cont_bbox": cont_bbox,
                    "cont_area": cont_area,
                    "blob_centroid": blob_centroid,
                    "surrounding_pixels": surrounding_pixels
                }

        return gray_img, contours_list, cleaned_blobs

    def _decolorate(self, full_img: np.ndarray[Any, Any]) -> np.ndarray[Any, np.dtype[np.uint8]]:
        """
        Elimina colores (rayones, resaltados, etc.) de la imagen, dejando solo blanco y negro.
        """
        # Máscara para píxeles negros (todos los canales <= threshold_black)
        mask_black = np.all(full_img < self.threshold_black, axis=2)
        
        # Máscara para píxeles blancos (todos los canales >= threshold_white)
        mask_white = np.all(full_img > self.threshold_white, axis=2)
        
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
        
    def compare_areas(self, metrics: Dict[str, Any], grey_img: np.ndarray[Any, Any]):
        cont_array_dict: Dict[int, Dict[str, Any]] = metrics.get("cont_array_dict", {})

        lines_cont: List[np.ndarray[Any, Any]] = []
        for pos, contours, in cont_array_dict.items():
            # cont_bbox = contours["cont_bbox"]
            cont_coords = contours["cont_coords"]
            # cont_area = contours["cont_area"]
            (_, _), (width, height), angle =  cv2.minAreaRect(cont_coords)

            if width < height:
                angle += 90
            angle = angle % 180.0

            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > 15:
                cv2.drawContours(image=grey_img, contours=[cont_coords], contourIdx = -1, color=[255, 255, 255], thickness=cv2.FILLED)
                logger.info(f"Linea: aspect_ratio: {aspect_ratio}")
                lines_cont.append(cont_coords)
            
            if aspect_ratio > 5 and angle < 11:
                cv2.drawContours(image=grey_img, contours=[cont_coords], contourIdx = -1, color=[255, 255, 255], thickness=cv2.FILLED)
                logger.info(f"Linea por angulo: {angle}°")
                lines_cont.append(cont_coords)
                
        return lines_cont, grey_img