import cv2
import numpy as np
import logging
from typing import Dict, Any, List
from core.factory.abstract_worker import PreprocessingAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons
from core.utils.image_analizer import extract_cc_metrics
from core.utils.image_utils import cropp_img
from services.output_service import save_shapes

logger = logging.getLogger(__name__)

class ImageRestorer(PreprocessingAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get("restorer", {})
        self.bin_interval = config["bin_interval"]
        self.kernel_threshold: int = self.worker_config.get("kernel_threshold", {})
        self.area_threshold: int = self.worker_config.get("area_threshold", {})
        self.output1 = config.get("contours")
        self.output2 = config.get("components")

    def preprocess(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        try:            
            if not manager.validate_cropped_img():
                logger.info(f"Sin cropped_img en el formatter")
                return False
                
            logger.debug("Polygonos revisados")
            polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
            if not polygons:
                return False
                        
            # 1. Analysis Phase
            for poly_id, polygon in polygons.items():
                logger.info(f"{poly_id}")
                # Acceso correcto a la imagen desde la dataclass
                cropped_img = polygon.cropped_img.cropped_img if polygon.cropped_img else None
                if cropped_img is None:
                    logger.warning(f"Imagen no encontrada para el polígono '{poly_id}'")
                    continue
               
                morph_stats = self.get_morph_stats(cropped_img.copy(), context, manager, poly_id)
                # restored = self._restore_morphology(morph_stats, cropped_img)

                return False
            
            return False

        except Exception as e:
            logger.error(f"Error restaurando: {e}", exc_info=True)
            return False

    def get_morph_stats(self, cropped_img: np.ndarray[Any, np.dtype[np.uint8]], context: Dict[str, Any], manager: DataFormatter, poly_id: str) -> List[Any]:
        metrics, bin_img = extract_cc_metrics(cropped_img, worker_config={}, binarice=True)
        if bin_img is None:
            return []
        img_w, img_h = bin_img.shape
        logger.info(f"{img_w, img_h}")
        cont_array_dict: Dict[int, Dict[str, Any]] = metrics["cont_array_dict"]
        noise_cont: Dict[int, Any] = {}
        blobs_cont: Dict[int, Any] = {}
        convex_list: List[Any] = []
        for pos, countours in cont_array_dict.items():
            cont_coords = countours["cont_coords"]
            convex_hull = countours["convex_hull"]
            hull_area = countours["hull_area"]
            cont_area = countours["cont_area"]
                
            if self.area_threshold >= cont_area:
                cont_bbox = countours["cont_bbox"]                
                # Crear una copia de la imagen para dibujar
                img_with_rect = bin_img.copy()
                
                # Asumir que cont_bbox es [x, y, width, height]
                if len(cont_bbox) == 4:
                    x, y, w, h = cont_bbox
                    cx = int((x + w) / 2)
                    cy = int((y + h) / 2)

                    moments = cv2.moments(cont_bbox)

                    logger.info(f"Momentos: {moments}")

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
                mean = np.mean(window)[0]
                white_pixels = np.sum(window >= self.bin_interval[1])
                black_pixels = np.sum(self.bin_interval[0] >= window)
                if not white_pixels and not black_pixels:
                    grey_pixels = window.size - white_pixels - black_pixels
                    logger.info(f"Grises: {grey_pixels}")

                logger.info(f"MEAN: {mean:.4f}, Blancos: {white_pixels}, Negros: {black_pixels}")

            convex_array = np.array(convex_hull).reshape(-1, 2)

            convex_array_reshape = np.delete(convex_array, 0, axis=0).astype(np.int32)
            convex_list.append({
                "convex_array_reshape": convex_array_reshape, 
                "cont_area": cont_area
            })
            # if cont_area == self.area_threshold:
            #     noise_cont[pos] = {
            #         "cont_area": cont_area, 
            #         "cont_coords":cont_coords, 
            #         "cont_bbox": cont_bbox,
            #         "convex_hull": convex_hull,
            #         "hull_area": hull_area
            #     }
            # else:
            #     blobs_cont[pos] = {
            #         "cont_area": cont_area, 
            #         "cont_coords":cont_coords, 
            #         "cont_bbox": cont_bbox,
            #         "convex_hull": convex_hull,
            #         "hull_area": hull_area
            #     }
            
        areas_sorted = sorted(convex_list, key=lambda x: x["cont_area"] ,reverse=True)
        # logger.info(f"Areas ordendas: {areas_sorted}")

        return []
        
        inter_area, _ = cv2.intersectConvexConvex(hull1, hull2)
        
        # logger.info(f"noise_cont: {noise_cont}, blobs_cont: {blobs_cont}")
        mapped_stats: np.ndarray[Any, np.dtype[np.uint16]] = metrics["mapped_stats"]
        bboxes_cc = np.column_stack([mapped_stats[:, 2], mapped_stats[:, 3], mapped_stats[:, 4], mapped_stats[:, 5]])
        
        mask = mapped_stats[:, 1] != self.area_threshold
        noise_cc = np.compress(mask, bboxes_cc, axis=0).astype(np.int32)

        cc = np.compress(mask, bboxes_cc, axis=0).astype(np.int32)

        # logger.info(f"cc: {cc}, noise_cc: {noise_cc}")
        

        if self.output2:
            from services.output_service import save_shapes
            worker_name = context.get("worker_name") or "restorer"
            image_name = manager.workflow.metadata.image_name if manager.workflow else ""
            output_paths = context["output_paths"]
    
            if self.output2:
                contours1: List[np.ndarray[Any, Any]] = []
                for item in noise_cont.values():
                    contour1 = np.array(item["cont_coords"], dtype=np.int32)
                    contours1.append(contour1)

                contours2: List[np.ndarray[Any, Any]] = []
                for item in blobs_cont.values():
                    contour2 = np.array(item["cont_coords"], dtype=np.int32)
                    contours2.append(contour2)

                # logger.info(f"Comparativa CONT: {contours2}")
                                    
                save_shapes(image_name, poly_id, cropped_img, output_paths, worker_name, contours1, contours2, method="contours")

            if self.output2:
                noise_cc_reshaped = noise_cc.reshape(-1, 2)
                cc_reshaped=cc.reshape(-1, 2)
                contours1: List[np.ndarray[Any, Any]]  = list(noise_cc)
                contours2: List[np.ndarray[Any, Any]]  = list(cc_reshaped)
                logger.info(f"SHAPES CC: {cc_reshaped.shape, noise_cc.shape}")
                # logger.info(f"Comparativa: CC: {cc}, Reshaped: {cc_reshaped}")

                cropped_image1 = cropp_img(bin_img, noise_cc)
                # cropped_image2 = cropp_img(bin_img, cc_reshaped)
                save_shapes(image_name, poly_id, cropped_image1, output_paths, worker_name, contours1, contours2, method="cc")
                # save_shapes(image_name, poly_id, cropped_image2, output_paths, worker_name, contours1, contours2, method="reshaped")

        return [noise_cc, cc, noise_cont, blobs_cont, bin_img]
    
    # def _restore_morphology(self, morph_stats: List[Any], cropped_img: np.ndarray[Any, np.dtype[np.uint8]]) -> np.ndarray[Any, np.dtype[np.uint8]]:
