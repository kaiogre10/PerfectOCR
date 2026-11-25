# core/workers/image_preparation/geometry_detector.py
import logging
import time
import numpy as np
from typing import Dict, Any, Optional, List
from app.models_builder import ModelsBuilder
from core.contracts.abstract_worker import ImagePrepAbstractWorker
from domain.data_formatter import DataFormatter
from utils.image_utils import binarice_img, morph_operations
from domain.buffer_handler import ArrayBuffer, ImageBuffer
from bidings import arrays
from utils.text_utils import set_ids
from services.output_service import save_croped_image

logger = logging.getLogger(__name__)

class GeometryDetector(ImagePrepAbstractWorker):
    """Detecta geometría con PaddleOCR"""
    __slots__ = ("kernel", "iterations", "output", "_engine", "bboxes_type")
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        worker_config = config.get("geometry_detect", {})
        self.kernel = worker_config["morph_kernel"]
        self.iterations = worker_config.get("iterations")
        self.bboxes_type = worker_config.get("bboxes_type", b"")
        self.centroids_type = worker_config.get("centroids_type", b"")
        self.padding = 1
        self.cropped_output = config.get("cropped_img")
        self.output = config.get("opened")
        self._engine = None
            
    @property
    def engine(self) -> Optional[Any]:
        if self._engine is None:
            paddle_manager = ModelsBuilder.get_instance()
            self._engine = paddle_manager.detection_engine

            if self._engine is None:
                logger.error("PaddleOCR DET no disponible")
                return None

        return self._engine
        
    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        try:
            engine = self.engine
            if engine is None:
                logger.error("Paddle DET no inicializado.")
                return False
            
            copy_ptr, full_imag = manager.get_img_copy()
            if full_imag is None or not copy_ptr:
                logger.error(f"No se pudo sacar la copia en el Formatter")
                return False
            
            full_imag = binarice_img(full_imag)

            full_imag = morph_operations(full_imag, self.kernel, self.iterations)

            polygons: List[List[List[float]]] = engine.ocr(img=full_imag, det=True, cls=False, rec=False)[0]
            if not manager.free_buffer(copy_ptr):
                raise RuntimeError("ERROR LIBERANDO COPIA EN MANAGER")

            if not polygons:
                logger.critical("No hay polygonos detectados")
                return False

            image_ptr = manager.get_img_ptr()

            if not image_ptr:
                raise AttributeError(f"ERROR EN EL PUNTERO: {image_ptr}")

            logger.debug(f"PUNTERO: {image_ptr}")

            time_buff = time.perf_counter()
            arr_ptr = manager.get_array_buffer(rows=len(polygons), cols=len(polygons[0]), channels=len(polygons[0][0]), dtype=self.bboxes_type)
            
            logger.debug(f"PUNTERO LIST PADDLE: {arr_ptr}")
            
            if not arr_ptr:
                logger.error(f"NO SE PUDO CREAR BUFFER PARA LOS BBOXES)")
                return False

            list_bbox = ArrayBuffer(arr_ptr, 3, self.bboxes_type).data
            
            np.copyto(dst=list_bbox, src=polygons, casting='unsafe')
            padding = self.padding
            c_centroids, c_bboxes, all_crops, crop_ptr = arrays.compute_polygons_geometry(arr_ptr, image_ptr, padding)
            if c_bboxes and c_centroids and crop_ptr:
                manager.delete_buffers([image_ptr, arr_ptr])    # Liberar imagen completa inmediatamente
            else:
                ptr_todel: List[int] = []
                if not c_bboxes:
                    ptr_todel.append(c_bboxes)
                if not c_centroids:
                    ptr_todel.append(c_centroids)
                if not all_crops:
                    ptr_todel.append(all_crops)
                if not crop_ptr:
                    ptr_todel.append(crop_ptr)

                if not manager.delete_buffers(ptr_todel + [image_ptr, arr_ptr]):
                    raise RuntimeError("NO SE PUDIERON LIBERAR LOS PUNTEROS DE GEOMETRIA")

            logger.info(f"PUNTEROS A LAS CROPPED: {crop_ptr}")

            list_centroids = ArrayBuffer(arr_ptr=c_centroids, dtype=self.centroids_type).data

            list_reshape_bbox = ArrayBuffer(arr_ptr=c_bboxes, dtype=self.bboxes_type).data

            logger.warning(f"TIEMPO BUFF_ARRAY: {time.perf_counter() - time_buff:.6f}'s:\n"f" {list_bbox.shape}")

            if self.cropped_output:
                image_name = manager.workflow.image_name
                for i, ptr in enumerate(crop_ptr):
                    poly_id = set_ids(i)
                    crop = ImageBuffer(ptr).data
                    save_croped_image(image_name, crop, poly_id)
            
            # np.maximum(0, all_bboxes[:, 0] - self.padding, out=all_bboxes[:, 0], dtype=self.bboxes_type)
            # np.maximum(0, all_bboxes[:, 1] - self.padding, out=all_bboxes[:, 1], dtype=self.bboxes_type)      ESto se queda por ser el layout original
            # np.minimum(img_cols, all_bboxes[:, 2] + self.padding, out=all_bboxes[:, 2], dtype=self.bboxes_type)
            # np.minimum(img_rows, all_bboxes[:, 3] + self.padding, out=all_bboxes[:, 3], dtype=self.bboxes_type)
            # valid_bbox_mask = (all_bboxes[:, 2] > all_bboxes[:, 0]) & (all_bboxes[:, 3] > all_bboxes[:, 1])
            #        
            polygons_list: List[Dict[str, Any]] = [{
                "bounding_box": list_reshape_bbox[i].tolist(),
                "centroid": list_centroids[i].tolist()
            }
            for i in range(list_reshape_bbox.shape[0])
            ]
            manager.create_polygon_dicts(polygons_list)
            if not manager.create_polygons([c_bboxes, c_centroids, all_crops, crop_ptr]):
                logger.critical("GeometryDetector: Fallo al estructurar polígonos.")
                return False

            else:
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
    #             image_name = manager.workflow.image_name if manager.workflow else ""
    #             # save_shapes(image_name, "not_valid", full_img, context["output_paths"], contours1, min_bboxes)
    #         return polygons_list
    #     except ValueError as e:
    #         logger.warning(f"Error de contornos: {e}", exc_info=True)
    #         return polygons_list
    