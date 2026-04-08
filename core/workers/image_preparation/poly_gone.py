# core/preprocessing/poly_gone.py
import numpy as np
import logging
import time
import dataclasses
from typing import Dict, Any, List, Tuple
from core.factory.abstract_worker import ImagePrepAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons
from core.utils.image_utils import  make_contiguous#, extract_contours_metrics
from services.output_service import save_croped_image#, save_shapes

logger = logging.getLogger(__name__)

class PolygonExtractor(ImagePrepAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        worker_config = config.get('polygon_extractor', {})
        self.bin_interval: Tuple[int, int] = config["bin_interval"]
        self.padding = worker_config.get("cropping_padding", 0.0)
        self.output = config.get("cropped_img", False)
        self.filtered_ouputs = config.get("final_polys", False)
        self.disoutput = config.get("discarded_polys", False)

    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """Extrae polígonos en batch usando operaciones vectorizadas para optimizar el recorte."""
        start_time = time.perf_counter()
        try:
            image_name = manager.workflow.metadata.image_name if manager.workflow else ""
            
            polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
                            
            if not polygons:
                logger.warning("PolygonExtractor: No se encontraron polígonos para procesar.")
                return False

            # 1. Fase de Análisis: Recopilar todas las bounding boxes
            poly_ids_order: List[str] = []
            all_bboxes: List[np.ndarray[Any, Any]] = []
            
            for poly_id, polygon in polygons.items():
                bbox = polygon.geometry.bounding_box
                if bbox.size != 4:
                    logger.warning(f"PolygonExtractor: Bounding box inválido para {poly_id}")
                    continue
                    
                # logger.info(f"{poly_id}: '{bbox}'")
                all_bboxes.append(bbox)
                poly_ids_order.append(poly_id)

            if not all_bboxes:
                logger.warning("PolygonExtractor: No hay bboxes válidos para procesar.")
                return False

            # 2. Fase de Decisión Vectorizada: Calcular todos los recortes con self.padding
            # pid = np.arange(len(poly_ids_order), dtype=np.int16)
            # bboxes_array = np.array(all_bboxes, np.int16)
            bboxes_array_id = np.column_stack([np.arange(len(poly_ids_order), dtype=np.int16), np.array(all_bboxes)])
            
            # Calcular coordenadas con self.padding usando operaciones vectorizadas
            img_h, img_w = manager.workflow.metadata.img_dims if manager.workflow else (0, 0)
            x1, y1, x2, y2 = bboxes_array_id[:, 1], bboxes_array_id[:, 2], bboxes_array_id[:, 3], bboxes_array_id[:, 4]

            # Aplicar self.padding y clipping de una vez
            px1 = np.maximum(0, x1 - self.padding)
            py1 = np.maximum(0, y1 - self.padding)
            px2 = np.minimum(img_w, x2 + self.padding)
            py2 = np.minimum(img_h, y2 + self.padding)

            # 3. Fase de Aplicación: Extraer imágenes solo para índices válidos
            valid_dims = (px2 > px1) & (py2 > py1)
            bboxes_array_id = np.compress(valid_dims, bboxes_array_id, 0)
            valid_indices = bboxes_array_id[:, 0].astype(np.int16)
            
            if valid_indices.size == 0:
                logger.warning("PolygonExtractor: No hay recortes válidos después del padding.")
                return False

            img_obj = manager.get_full_img()
            full_img: np.ndarray[Any, np.dtype[np.uint8]] | None = img_obj.full_img if img_obj is not None else None
            if full_img is None:
                logger.error(f"No Hay full_img en el Formatter")
                return False
            logger.debug("Full_img obtenida con éxito")
            # Liberar la imagen completa lo antes posible
            manager.update_full_img(corrected=False, full_img=None)

            # full_img = make_contiguous(full_img)
            # contours_list, metrics = extract_contours_metrics(full_img)

            # asigned_cont = self.asign_contours(metrics[:, [0, 5, 6, 18, 19, 20, 21]], bboxes_array_id)
            # valid_cont = asigned_cont[:, 0].tolist()
            # all_contour_indices = metrics[:, 0].astype(int)
            # contours = [contours_list[i][1] for i in valid_cont]
            # contours2 = [contours_list[i][1] for i in all_contour_indices if i not in valid_cont]
            # save_shapes(image_name, "not_valid", full_img, context["output_paths"], contours, contours2)
            cropped_images: Dict[str, np.ndarray[Any, np.dtype[np.uint8]]] = {}
            discarded_poly_ids: List[str] = []
            valid_poly_ids: List[str] = []
            valid_polygons_data: List[Dict[str, Any]] = []

            for idx in valid_indices:
                poly_id = poly_ids_order[idx]
                poly_index = idx
                crop_x1, crop_y1 = int(px1[idx]), int(py1[idx])
                crop_x2, crop_y2 = int(px2[idx]), int(py2[idx])
                cropped: np.ndarray[Any, np.dtype[np.uint8]] = make_contiguous(full_img[crop_y1:crop_y2, crop_x1:crop_x2])

                if self.output:
                    self.save_debug(cropped, context, manager, "all", poly_id) # type: ignore

                poly_mean = int(np.mean(cropped))
                if not bool(self.bin_interval[0] < poly_mean < self.bin_interval[1]):
                    discarded_poly_ids.append(poly_id)
                    # logger.info(f"ELIMINADO '{poly_id}': FUERA DE RANGO DE COLOR")
                    if self.disoutput:
                        status = "bn"
                        pid = f"{poly_id}_{poly_mean}"
                        self.save_debug(cropped, context, manager, status, pid)
                
                else:
                    valid_poly_ids.append(poly_id)
                    valid_polygons_data.append({
                    "poly_id": poly_id,
                    "poly_index": poly_index,
                    "cropped": make_contiguous(cropped),
                    })

            full_img = None
            if not valid_polygons_data:
                logger.warning("PolygonExtractor: No hay polígonos válidos para procesar.")
                return False

            for i, p_data in enumerate(valid_polygons_data):
                new_id = f"poly_{i:04d}"
                cropped_images[new_id] = p_data["cropped"]

            # Eliminar los descartados del manager.workflow.polygons
            for poly_id in discarded_poly_ids:
                pid = poly_id.split(" ")[0]

                if pid in manager.workflow.polygons if manager.workflow else None:
                    del manager.workflow.polygons[pid] # type: ignore

                pid = poly_id.split(" ")[0]
                if pid in manager.workflow.polygons if manager.workflow else None:
                    del manager.workflow.polygons[pid]# type: ignore

            # Reindexar los polígonos válidos en el manager
            new_polygons: Dict[str, Polygons] = {}
            for idx, old_id in enumerate(valid_poly_ids):

                if old_id not in manager.workflow.polygons:
                    continue

                new_id = f"poly_{idx:04d}"
                poly_index = idx
                poly_obj = manager.workflow.polygons[old_id] # type: ignore
                poly_obj = dataclasses.replace(poly_obj, polygon_id=new_id, poly_index=poly_index)
                new_polygons[new_id] = poly_obj
                    
            manager.workflow.polygons = new_polygons
            
            if not manager.save_cropped_images(cropped_images):
                logger.error("No se pudieron guardar las imagenes en el manager")
                return False

            # logger.info(f"'{len(cropped_images)}' polígonos recortados en {time.perf_counter() - start_time:.6f}s.")

            if self.filtered_ouputs:
                polygons = manager.workflow.polygons if manager.workflow else {}

                for poly_id, polygon in polygons.items():
                    cropped_img = polygon.cropped_img.cropped_img if polygon.cropped_img else None
                    self.save_debug(cropped_img, context, manager, "filtered", poly_id)
            return True

        except Exception as e:
            logger.error(f"Error en PolygonExtractor: {e}", exc_info=True)
        return False
    
    def asign_contours(self, metrics: np.ndarray[Any, Any], bboxes_array: np.ndarray[Any, Any]) -> np.ndarray[Any, np.dtype[np.int16]]:
        # Índice original de cada contorno
        contour_ids = metrics[:, 0]

        c_x = metrics[:, 1][:, None]
        c_y = metrics[:, 2][:, None]

        x_min = metrics[:, 3][:, None]
        x_max = metrics[:, 4][:, None]
        y_min = metrics[:, 5][:, None]
        y_max = metrics[:, 6][:, None]

        bbox_ids = bboxes_array[:, 0]

        b_x1 = bboxes_array[:, 1][None, :]
        b_y1 = bboxes_array[:, 2][None, :]
        b_x2 = bboxes_array[:, 3][None, :]
        b_y2 = bboxes_array[:, 4][None, :]

        centroid_inside = (c_x > b_x1) & (c_x < b_x2) & (c_y > b_y1) & (c_y < b_y2)

        # 2) contorno completamente dentro (evita ruido parcialmente metido)
        contour_fully_inside = (x_min > b_x1) & (x_max < b_x2) & (y_min > b_y1) & (y_max < b_y2)

        # criterio final
        is_inside = centroid_inside & contour_fully_inside

        has_bbox = np.any(is_inside, axis=1)          # (N,)
        bbox_local_idx = np.argmax(is_inside, axis=1)  # (N,)

        # Mapear bbox local -> bbox original
        bbox_original_ids = np.where(has_bbox, bbox_ids[bbox_local_idx], -1)

        # Resultado final: [contour_original_idx, bbox_original_idx]
        mapped_ids = np.column_stack([contour_ids, bbox_original_ids]).astype(np.int16)
        return np.compress(mapped_ids[:, 1] >= 0, mapped_ids, 0)
    
    def save_debug(self, polygon: np.ndarray[Any, np.dtype[np.uint8]], context: Dict[str, Any], manager: DataFormatter, status: str, id: str):
        image_name = manager.workflow.metadata.image_name if manager.workflow else ""
        worker_name = context.get("worker_name") or "poly_gone"
        output_paths = context["output_paths"]
        img_id = f"{status}_{id}_close"
        save_croped_image(image_name, img_id, polygon, output_paths, worker_name)