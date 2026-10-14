# core/preprocessing/poly_gone.py
import numpy as np
import logging
import dataclasses
from typing import Dict, Any
from core.contracts.abstract_worker import ImagePrepAbstractWorker
from domain.data_formatter import DataFormatter
from utils.image_utils import validate_image
from services.output_service import save_croped_image
from domain.class_models import StringsModels

logger = logging.getLogger(__name__)

class PolygonExtractor(ImagePrepAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        worker_config = config.get('polygon_extractor', {})
        self.padding = worker_config.get("cropping_padding")
        self.output = config.get("cropped_img", False)
        self.filtered_ouputs = config.get("final_polys", False)
        self.disoutput = config.get("discarded_polys", False)

    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """Extrae, filtra y actualiza polígonos en un solo paso, optimizando el proceso."""
        try:
            if not manager.workflow or not manager.workflow.polygons:
                logger.warning("PolygonExtractor: No hay workflow o polígonos para procesar.")
                return False

            polygons = manager.workflow.polygons

            # 1. Recopilar Bounding Boxes y IDs
            poly_ids_order = list(polygons.keys())
            all_bboxes = np.asarray([polygons[pid].bounding_box for pid in poly_ids_order], dtype=np.int32, order='C').reshape(-1, 4, order='C')

            if all_bboxes.size < 1 or all_bboxes.size % 4 != 0:
                logger.warning("PolygonExtractor: No hay bboxes válidos para procesar.")
                return False

            full_img = manager.get_full_img()
            if full_img is None:
                logger.error("No se pudo obtener full_img del Formatter.")
                return False

            if full_img.size < 2:
                logger.error("SIZE MENOR A 1")
                return False

            # 2. Calcular Coordenadas de Recorte con Padding
            img_h, img_w = full_img.shape[:2]
            x1, y1, x2, y2 = all_bboxes[:, 0], all_bboxes[:, 1], all_bboxes[:, 2], all_bboxes[:, 3]

            # size_bbox = np.asarray((x2 - x1) * (y2 - y1), dtype=np.int32, order='C',)
            # bboxes = np.column_stack((all_bboxes, size_bbox))
            # logger.info(f"\n{np.array2string(bboxes)}")
            # arr_ptr = bboxes.ctypes.data
            # arr_dims = bboxes.ndim
            # arr_shape = bboxes.shape
            # arr_size = bboxes.size
            # arr_itemsize = bboxes.itemsize
            # arr_metadata = (arr_ptr, arr_dims, arr_shape, arr_size, arr_itemsize)

            px1 = np.maximum(0, x1 - self.padding, dtype=np.int32)
            py1 = np.maximum(0, y1 - self.padding, dtype=np.int32)
            px2 = np.minimum(img_w, x2 + self.padding, dtype=np.int32)
            py2 = np.minimum(img_h, y2 + self.padding, dtype=np.int32)

            valid_bbox_mask = (px2 > px1) & (py2 > py1)
            valid_bbox_indices = np.flatnonzero(valid_bbox_mask)

            for invalid_idx in np.flatnonzero(~valid_bbox_mask):
                logger.debug(f"POLÍGONO INVÁLIDO: {poly_ids_order[invalid_idx]}")

            # 3. Bucle Único de Filtrado, Actualización y Re-indexado
            new_polygons: Dict[str, Any] = {}
            cropped_images_to_save: Dict[str, np.ndarray[Any, np.dtype[np.uint8]]] = {}
            new_poly_idx = 0

            for idx in valid_bbox_indices:
                old_poly_id = poly_ids_order[idx] # type: ignore
                crop_x1, crop_y1 = px1[idx], py1[idx]
                crop_x2, crop_y2 = px2[idx], py2[idx]

                # Recortar la imagen
                cropped = np.ascontiguousarray(np.asarray(full_img[crop_y1:crop_y2, crop_x1:crop_x2].copy(), order='C'), dtype=np.uint8)

                # Filtrar por intervalo de color
                if not validate_image(cropped):
                    logger.debug(f"FILTRADO POR MEDIA DE COLOR: {old_poly_id}")
                    if self.disoutput:
                        pid = f"{old_poly_id}_B/N"
                        self.save_debug(cropped, manager, "bn_discarded", pid)
                    continue

                # Si el polígono es válido, se crea el nuevo objeto completo
                new_id = f"{StringsModels.POLYS_IDS.value}{new_poly_idx:04d}"
                
                # Actualizar geometría con las nuevas coordenadas (con padding)
                original_poly = polygons[old_poly_id]
                original_bbox = original_poly.bounding_box

                # Crear el nuevo objeto Polygons actualizado y añadirlo al diccionario final
                new_polygons[new_id] = dataclasses.replace(
                    original_poly,
                    polygon_id=new_id,
                    poly_index=new_poly_idx,
                    bounding_box=original_bbox
                )
                cropped_images_to_save[new_id] = cropped
                new_poly_idx += 1

                if self.output:
                    self.save_debug(cropped, manager, "all_valid", new_id)

            if not manager.free_buffer():
                logger.error("NO SE PUDO LIBERAR FULL IMAGE")

            # 4. Actualización Final y Limpia en el Manager
            if not new_polygons:
                logger.warning("PolygonExtractor: Ningún polígono superó los filtros.")
                manager.workflow.polygons = {}
                return False

            manager.workflow.polygons = new_polygons
            
            if not manager.save_cropped_images(cropped_images_to_save):
                logger.error("No se pudieron guardar las imágenes recortadas en el manager")
                return False

            # logger.debug(f"'{len(new_polygons)}' polígonos extraídos y filtrados en {time.perf_counter() - start_time:.6f}s.")

            if self.filtered_ouputs:
                for poly_id, polygon in new_polygons.items():
                    if polygon.cropped_img is not None:
                        self.save_debug(polygon.cropped_img, manager, "filtered_final", poly_id)
            return True

        except Exception as e:
            logger.error(f"Error en PolygonExtractor: {e}", exc_info=True)
            return False
    
    def save_debug(self, polygon: np.ndarray[Any, np.dtype[np.uint8]], manager: DataFormatter, status: str, id: str):
        image_name = manager.workflow.image_name if manager.workflow else ""
        img_id = f"{status}_{id}_close"
        save_croped_image(image_name, img_id, polygon)