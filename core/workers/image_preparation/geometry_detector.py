# core/workers/image_preparation/geometry_detector.py
import logging
import cv2
# import time
import numpy as np
from typing import Dict, Any, Optional, List
from app.models_manager import ModelsManager
from core.factory.abstract_worker import ImagePrepAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.utils.image_utils import binarice_img, make_contiguous, cropp_img # get_contours_values
from services.output_service import save_croped_image

logger = logging.getLogger(__name__)

class GeometryDetector(ImagePrepAbstractWorker):
    """
    Detecta geometría con PaddleOCR:
    """
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        worker_config = config.get("geometry_detect", {})
        kernel_threshold = worker_config["morph_kernel"]
        self.kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_threshold[0], kernel_threshold[1]))
        self.iterations = worker_config.get("iterations")
        self.output = config.get("deleted_polys")
        self.output2 = config.get("opened")
        self._engine = None
            
    @property
    def engine(self) -> Optional[Any]:
        if self._engine is None:
            paddle_manager = ModelsManager.get_instance()
            self._engine = paddle_manager.detection_engine

            if self._engine is None:
                logger.error("GeometryDetector: Motor de detección no disponible en PaddleManager")
        
            logger.debug("GeometryDetector: Motor de detección obtenido del PaddleManager")
    
        return self._engine
        
    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        # start_time = time.perf_counter()
        worker_name = context.get("worker_name") or "geometry_detector"
        try:
            engine = self.engine
            if engine is None:
                logger.error("PaddleOCR no inicializado.")
                return False

            img_obj = manager.get_full_img()
            full_imag = img_obj.full_img if img_obj is not None else None
            
            if full_imag is None:
                logger.error(f"No Hay full_img en el Formatter")
                return False

            bin_img = binarice_img(full_imag, {})

            img = make_contiguous(cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, self.kernel, iterations=self.iterations))

            if self.output2:
                image_name = manager.workflow.metadata.image_name if manager.workflow else ""
                # imag_id = f"opened_{image_name}_{worker_name}"
                img_id = f"bin_img_{image_name}_{worker_name}_+1"
                save_croped_image(image_name, img_id, img, worker_name)
                # save_croped_image(image_name, imag_id, img, output_paths, worker_name)
                
            # paddle_time = time.perf_counter()
            polygons = engine.ocr(img=img, det=True, cls=False, rec=False)[0]
            # logger.info(f"Tiempo de detección de paddle: {time.perf_counter() - paddle_time:.6f}'s")
            if not polygons:
                logger.critical("No hay polygonos detectados")
                return False
            
            # total_conts = len(polygons)
            # geometry_array = np.zeros((total_conts, 6), np.float32)
            # geometry_array = np.zeros((total_conts, 17), np.float32)

            polygons_list: List[Dict[str, Any]] = []
            for idx, poly_pts in enumerate(polygons):
                poly_id = f"poly_{idx:04d}"
                coords = np.array([[p[0], p[1]] for p in poly_pts], np.float32)
                bbox = np.array([coords[:, 0].min(), coords[:, 1].min(), coords[:, 0].max(), coords[:, 1].max()], np.float32)
                centroid = np.mean(coords, axis=0, dtype=np.float32)

                # geometry_array[idx, [0, 1, 2, 3, 4, 5]] = bbox[0], bbox[1], bbox[2], bbox[3], centroid[0], centroid[1]
                if self.output:
                    cropped = cropp_img(img, bbox)
                    worker_name = context.get("worker_name") or "geometry_detector"
                    pid = f"{poly_id}_{worker_name}"
                    image_name = manager.workflow.metadata.image_name if manager.workflow else ""
                    save_croped_image(image_name, pid, cropped, worker_name)
            
                polygons_list.append({
                    "poly_index": idx,
                    "polygon_coords": coords.reshape(-1, 1, 2),
                    "bounding_box": bbox,
                    "centroid": centroid,
                })

            # ind = np.arange(total_conts)
            # logger.info("POLYS ARRAY:\n"f"{np.array2string(np.column_stack([ind, geometry_array]), suppress_small=True)}")

            # final_polygons_list = self.validate_polygons(img, polygons_list, manager, context)
            final_polygons: Dict[str, Dict[str, Any]] = {}
            for new_idx, poly_data in enumerate(polygons_list):
                poly_id = f"poly_{new_idx:04d}"
                final_polygons[poly_id] = poly_data

          #  logger.info(f"FINAL: {len(final_polygons)}")

            if not manager.create_polygon_dicts(final_polygons):
                logger.critical("GeometryDetector: Fallo al estructurar polígonos.")
                return False

            else:
                # logger.info(f"{len(final_polygons)} poligonos válidos detectados en: {time.perf_counter()-start_time:.6f}s")
                return True
        
        except Exception as e:
            logger.critical(f"Error en procesamiento vectorizado de geometría: {e}", exc_info=True)
        return False

    # def validate_polygons(self, img: np.ndarray[Any, np.dtype[np.uint8]], polygons_list: List[Dict[str, Any]], manager: DataFormatter, context: Dict[str, Any]) -> List[Dict[str, Any]]:
    #     try:
    #         # _, metrics = get_contours_values(img)
    #         all_bboxes = np.array([p["bounding_box"] for p in polygons_list])
    #         bbox_ids = np.arange(len(polygons_list), dtype=np.int16)
    #         bboxes_array = np.column_stack([bbox_ids, np.array(all_bboxes)])

    #         # Índice original de cada contorno
    #         contour_ids = metrics[:, 0]
            
    #         c_x = metrics[:, 5][:, None]
    #         c_y = metrics[:, 6][:, None]

    #         x_min = metrics[:, 18][:, None]
    #         x_max = metrics[:, 19][:, None]
    #         y_min = metrics[:, 20][:, None]
    #         y_max = metrics[:, 21][:, None]

    #         b_x1 = bboxes_array[:, 1][None, :]
    #         b_y1 = bboxes_array[:, 2][None, :]
    #         b_x2 = bboxes_array[:, 3][None, :]
    #         b_y2 = bboxes_array[:, 4][None, :]

    #         # cont_h = y_max - y_min
    #         # bbox_h = b_y2 - b_y1

    #         # mask_h = (cont_h > bbox_h)
    #         # invalid_mask = np.any(mask_h, axis=1)
    #         # invalid_ids = contour_ids[invalid_mask].astype(np.int16)

    #         centroid_inside = (c_x > b_x1) & (c_x < b_x2) & (c_y > b_y1) & (c_y < b_y2)

    #         # 2) contorno completamente dentro (evita ruido parcialmente metido)
    #         contour_fully_inside = (x_min > b_x1) & (x_max < b_x2) & (y_min > b_y1) & (y_max < b_y2)

    #         # criterio final
    #         is_inside = centroid_inside & contour_fully_inside

    #         has_bbox = np.any(is_inside, axis=1)          # (N,)
    #         bbox_local_idx = np.argmax(is_inside, axis=1)  # (N,)

    #         # Mapear bbox local -> bbox original
    #         bbox_original_ids = np.where(has_bbox, bbox_ids[bbox_local_idx], -1)

    #         # Resultado final: [contour_original_idx, bbox_original_idx]
    #         mapped = np.column_stack([contour_ids, bbox_original_ids]).astype(np.int16)

    #         # inv_mapped_ids = np.compress(mapped[:, 1] < 0, mapped, 0) 
    #         mapped_ids = np.compress(mapped[:, 1] >= 0, mapped, 0)

    #         conts_bbbox = np.bincount(mapped_ids[:, 1])
    #         # logger.info("Contornos por bbox:"
    #         #             "\n"f"{conts_bbbox}")
            
    #         for poly in polygons_list:
    #             idx = int(poly["poly_index"])
    #             poly["contours_count"] = int(conts_bbbox[idx])
    #         # min_indx = np.where(conts_bbbox < 1)[0]

    #         # img_obj = manager.get_full_img()
    #         # full_img = img_obj.full_img if img_obj is not None else None
    #         # if full_img is None:
    #         #     logger.error(f"No Hay full_img en el Formatter")
    #         #     return polygons_list
            
    #         # min_bboxes = [p["polygon_coords"] for p in polygons_list if p["poly_index"] in min_indx]
            
    #         # inv_valid_cont = [int(idx) for idx in inv_mapped_ids[:, 0].tolist()]
    #         # valid_cont = [int(idx) for idx in mapped_ids[:, 0].tolist()]

    #         # contours1 = [contours_list[i][1] for i in inv_valid_cont]
    #         # contours2 = [contours_list[i][1] for i in valid_cont]

    #         # bboxes = [p["polygon_coords"] for p in polygons_list]
    #         # contours2.extend(bboxes)

    #         if self.output:
    #             image_name = manager.workflow.metadata.image_name if manager.workflow else ""
    #             # save_shapes(image_name, "not_valid", full_img, context["output_paths"], contours1, min_bboxes)
    #         return polygons_list
    #     except ValueError as e:
    #         logger.warning(f"Error de contornos: {e}", exc_info=True)
    #         return polygons_list