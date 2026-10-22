# core/preprocessing/poly_gone.py
import numpy as np
import logging
from typing import Dict, Any, List
from core.contracts.abstract_worker import ImagePrepAbstractWorker
from domain.data_formatter import DataFormatter
from utils.image_utils import validate_image
from services.output_service import save_croped_image

logger = logging.getLogger(__name__)

class PolygonExtractor(ImagePrepAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
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
            poly_ids_order: List[str] = list(polygons.keys())
            total_polys = len(polygons)

            list_bbox = [polygons[pid].bounding_box for pid in poly_ids_order]

            full_img = manager.get_full_img()
            if full_img is None:
                logger.error("No se pudo obtener full_img del Formatter.")
                return False
            
            # 3. Bucle Único de Filtrado, Actualización y Re-indexado
            final_polygons: Dict[str, Any] = {}

            for idx, bbox in enumerate(list_bbox):
                # Recortar la imagen
                cropped = np.ascontiguousarray(np.asarray(full_img[bbox[1]:bbox[3], bbox[0]:bbox[2]].copy(), order='C'), dtype=np.uint8)

                # Filtrar por intervalo de color
                poly_id = poly_ids_order[idx]
                if not validate_image(cropped):
                    logger.debug(f"FILTRADO POR MEDIA DE COLOR: {poly_id}")
                    if self.disoutput:
                        pid = f"{poly_id}_B/N"
                        self.save_debug(cropped, manager, "bn_discarded", pid)
                    continue

                if self.output:
                    self.save_debug(cropped, manager, "all_valid", poly_id) # type: ignore

                final_polygons[poly_id] = cropped

            img_ptr = manager.get_img_ptr()
            
            if not manager.free_buffer(img_ptr):
                logger.error("NO SE PUDO LIBERAR FULL IMAGE")
                return False

            # 4. Actualización Final y Limpia en el Manager
            if not final_polygons:
                logger.warning("PolygonExtractor: Ningún polígono superó los filtros.")
                manager.workflow.polygons = {}
                return False
            
            if len(final_polygons) != total_polys:
                reindex = False
            else:
                reindex = True
            
            if not manager.save_cropped_images(final_polygons, reindex):
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