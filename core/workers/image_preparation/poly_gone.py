# core/preprocessing/poly_gone.py
import numpy as np
import logging
from typing import Dict, Any, List
from core.contracts.abstract_worker import ImagePrepAbstractWorker
from domain.data_formatter import DataFormatter
from utils.image_utils import validate_image
from services.output_service import save_croped_image
from domain.buffer_handler import ArrayBuffer

logger = logging.getLogger(__name__)

class PolygonExtractor(ImagePrepAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        worker_config = config.get('polygon_extractor', {})
        self.padding = worker_config.get("cropping_padding")
        self.bboxes_type = worker_config.get("bboxes_type")
        self.output = config.get("cropped_img", False)
        self.filtered_ouputs = config.get("final_polys", False)
        self.disoutput = config.get("discarded_polys", False)
        self.dims = 2

    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """Extrae, filtra y actualiza polígonos en un solo paso, optimizando el proceso."""
        try:
            if not manager.workflow or not manager.workflow.polygons:
                logger.warning("PolygonExtractor: No hay workflow o polígonos para procesar.")
                return False

            polygons = manager.workflow.polygons

            # 1. Recopilar Bounding Boxes y IDs
            poly_ids_order: List[str] = list(polygons.keys())
            total_bboxes = len(poly_ids_order)
            logger.info(f"TOTAL BBOXES: {total_bboxes}")

            bbxes_ptr = manager.get_array_buffer(rows=total_bboxes, cols=4, channels=1, dtype=self.bboxes_type, dims=self.dims)
            if not bbxes_ptr:
                logger.error(f"NO SE PUDO CREAR BUFFER PARA LOS BBOXES)")
                return False
            
            bboxes = np.array([polygons[pid].bounding_box for pid in poly_ids_order], dtype=np.int32).reshape(-1, 4, order='C')
            all_bboxes = ArrayBuffer(bbxes_ptr, self.dims).data

            np.copyto(dst=all_bboxes, src=bboxes, casting='no')
            del bboxes

            if all_bboxes.size < 3 or all_bboxes.size % 4 != 0:
                logger.warning("PolygonExtractor: No hay bboxes válidos para procesar.")
                return False

            full_img = manager.get_full_img()
            if full_img is None:
                logger.error("No se pudo obtener full_img del Formatter.")
                return False

            # 2. Calcular Coordenadas de Recorte con Padding
            img_h, img_w = full_img.shape

            np.maximum(0, all_bboxes[:, 0] - self.padding, out=all_bboxes[:, 0])
            np.maximum(0, all_bboxes[:, 1] - self.padding, out=all_bboxes[:, 1])
            np.minimum(img_w, all_bboxes[:, 2] + self.padding, out=all_bboxes[:, 2])
            np.minimum(img_h, all_bboxes[:, 3] + self.padding, out=all_bboxes[:, 3])

            valid_bbox_mask = (all_bboxes[:, 2] > all_bboxes[:, 0]) & (all_bboxes[:, 3] > all_bboxes[:, 1])

            # valid_bbox_indices = np.compress(valid_bbox_mask, all_bboxes, axis=0)
            # crops = np.empty(all_bboxes.shape, dtype=np.uint8, order='C')
            # np.array_split(full_img[all_bboxes[:, 1]:all_bboxes[:, 3], all_bboxes[:, 0]:all_bboxes[:, 2]].copy(order='C'))
            # crops = np.select(condlist=valid_bbox_mask, choicelist=full_img[all_bboxes[:, 1]:all_bboxes[:, 3], all_bboxes[:, 0]:all_bboxes[:, 2]].copy())
            valid_bbox_indices = np.flatnonzero(valid_bbox_mask)
            
            # for invalid_idx in np.flatnonzero(~valid_bbox_mask):
            #     logger.debug(f"POLÍGONO INVÁLIDO: {poly_ids_order[invalid_idx]}")

            # 3. Bucle Único de Filtrado, Actualización y Re-indexado
            final_polygons: Dict[str, Any] = {}

            for _, idx in enumerate(valid_bbox_indices):
                # x1, y1, x2, y2 = all_bboxes[:, 0], all_bboxes[:, 1], all_bboxes[:, 2], all_bboxes[:, 3]
                crop_x1, crop_y1 = all_bboxes[idx, 0], all_bboxes[idx, 1]
                crop_x2, crop_y2 = all_bboxes[idx, 2], all_bboxes[idx, 3]

                # Recortar la imagen
                cropped = np.ascontiguousarray(np.asarray(full_img[crop_y1:crop_y2, crop_x1:crop_x2].copy(), order='C'), dtype=np.uint8)

                # Filtrar por intervalo de color
                old_poly_id = poly_ids_order[idx] # type: ignore
                if not validate_image(cropped):
                    logger.debug(f"FILTRADO POR MEDIA DE COLOR: {old_poly_id}")
                    if self.disoutput:
                        pid = f"{old_poly_id}_B/N"
                        self.save_debug(cropped, manager, "bn_discarded", pid)
                    continue

                if self.output:
                    self.save_debug(cropped, manager, "all_valid", old_poly_id) # type: ignore

                final_polygons[old_poly_id] = cropped

            if not manager.free_buffer():
                logger.error("NO SE PUDO LIBERAR FULL IMAGE")
                return False

            # 4. Actualización Final y Limpia en el Manager
            if not final_polygons:
                logger.warning("PolygonExtractor: Ningún polígono superó los filtros.")
                manager.workflow.polygons = {}
                return False
            
            if not manager.save_cropped_images(final_polygons):
                logger.error("No se pudieron guardar las imágenes recortadas en el manager")
                return False

            logger.debug(f"'{len(final_polygons)}' polígonos extraídos y filtrados.")

            return True

        except Exception as e:
            logger.error(f"Error en PolygonExtractor: {e}", exc_info=True)
            return False
    
    def save_debug(self, polygon: np.ndarray[Any, np.dtype[np.uint8]], manager: DataFormatter, status: str, id: str):
        image_name = manager.workflow.image_name if manager.workflow else ""
        img_id = f"{status}_{id}_close"
        save_croped_image(image_name, img_id, polygon)