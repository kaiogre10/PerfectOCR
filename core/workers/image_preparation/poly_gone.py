# core/preprocessing/poly_gone.py
import numpy as np
import logging
import time
import dataclasses
from typing import Dict, Any, Tuple
from core.factory.abstract_worker import ImagePrepAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons
from core.utils.image_utils import  make_contiguous
from services.output_service import save_croped_image

logger = logging.getLogger(__name__)

class PolygonExtractor(ImagePrepAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        worker_config = config.get('polygon_extractor', {})
        self.bin_interval: Tuple[float, float] = config["bin_interval"]
        self.padding = worker_config.get("cropping_padding", 0.0)
        self.output = config.get("cropped_img", False)
        self.filtered_ouputs = config.get("final_polys", False)
        self.disoutput = config.get("discarded_polys", False)

    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """Extrae, filtra y actualiza polígonos en un solo paso, optimizando el proceso."""
        start_time = time.perf_counter()
        try:
            if not manager.workflow or not manager.workflow.polygons:
                logger.warning("PolygonExtractor: No hay workflow o polígonos para procesar.")
                return False

            polygons: Dict[str, Polygons] = manager.workflow.polygons

            # 1. Recopilar Bounding Boxes y IDs
            poly_ids_order = list(polygons.keys())
            all_bboxes = np.array([polygons[pid].geometry.bounding_box for pid in poly_ids_order], dtype=np.int16)

            if not all_bboxes.size:
                logger.warning("PolygonExtractor: No hay bboxes válidos para procesar.")
                return False

            # 2. Calcular Coordenadas de Recorte con Padding (Vectorizado)
            img_h, img_w = manager.workflow.metadata.img_dims
            x1, y1, x2, y2 = all_bboxes[:, 0], all_bboxes[:, 1], all_bboxes[:, 2], all_bboxes[:, 3]
            px1 = np.maximum(0, x1 - self.padding)
            py1 = np.maximum(0, y1 - self.padding)
            px2 = np.minimum(img_w, x2 + self.padding)
            py2 = np.minimum(img_h, y2 + self.padding)

            # 3. Obtener Imagen y Liberar Memoria del Manager
            img_obj = manager.get_full_img()
            full_img = img_obj.full_img if img_obj else None
            if full_img is None:
                logger.error("No se pudo obtener full_img del Formatter.")
                return False
            manager.update_full_img(corrected=False, full_img=None)
            full_img = make_contiguous(full_img)

            # 4. Bucle Único de Filtrado, Actualización y Re-indexado
            new_polygons: Dict[str, Polygons] = {}
            cropped_images_to_save: Dict[str, np.ndarray[Any, np.dtype[np.uint8]]] = {}
            new_poly_idx = 0

            for idx, old_poly_id in enumerate(poly_ids_order):
                # Filtrar por dimensiones válidas
                if not ((px2[idx] > px1[idx]) or (py2[idx] > py1[idx])):
                    logger.info(f"POLÍGONO INVÁLIDO: {old_poly_id}")
                    # if self.disoutput:
                    #     pid = f"{old_poly_id}"
                    #     self.save_debug(cropped, context, manager, "invalid_dims", pid)
                    continue

                # Recortar la imagen
                crop_x1, crop_y1, crop_x2, crop_y2 = int(px1[idx]), int(py1[idx]), int(px2[idx]), int(py2[idx])
                cropped = np.ascontiguousarray(full_img[crop_y1:crop_y2, crop_x1:crop_x2], dtype=np.uint8)

                # Filtrar por intervalo de color
                poly_mean = np.mean(cropped)
                if not (self.bin_interval[0] < poly_mean < self.bin_interval[1]):
                    # if self.disoutput:
                    #     pid = f"{old_poly_id}_{poly_mean}"
                    #     self.save_debug(cropped, context, manager, "bn_discarded", pid)
                    continue

                # Si el polígono es válido, se crea el nuevo objeto completo
                new_id = f"poly_{new_poly_idx:04d}"
                
                # Actualizar geometría con las nuevas coordenadas (con padding)
                new_bbox = np.array([crop_x1, crop_y1, crop_x2, crop_y2])
                original_poly = polygons[old_poly_id]
                new_geometry = dataclasses.replace(original_poly.geometry, bounding_box=new_bbox)

                # Crear el nuevo objeto Polygons actualizado y añadirlo al diccionario final
                new_polygons[new_id] = dataclasses.replace(
                    original_poly,
                    polygon_id=new_id,
                    poly_index=new_poly_idx,
                    geometry=new_geometry
                )
                cropped_images_to_save[new_id] = cropped
                new_poly_idx += 1

                if self.output:
                    self.save_debug(cropped, context, manager, "all_valid", new_id)

            # 5. Actualización Final y Limpia en el Manager
            if not new_polygons:
                logger.warning("PolygonExtractor: Ningún polígono superó los filtros.")
                manager.workflow.polygons = {}
                return False

            manager.workflow.polygons = new_polygons
            
            if not manager.save_cropped_images(cropped_images_to_save):
                logger.error("No se pudieron guardar las imágenes recortadas en el manager")
                return False

            # logger.info(f"'{len(new_polygons)}' polígonos extraídos y filtrados en {time.perf_counter() - start_time:.6f}s.")

            if self.filtered_ouputs:
                for poly_id, polygon in new_polygons.items():
                    if polygon.cropped_img and polygon.cropped_img.cropped_img is not None:
                        self.save_debug(polygon.cropped_img.cropped_img, context, manager, "filtered_final", poly_id)
            return True

        except Exception as e:
            logger.error(f"Error en PolygonExtractor: {e}", exc_info=True)
            return False
    
    def save_debug(self, polygon: np.ndarray[Any, np.dtype[np.uint8]], context: Dict[str, Any], manager: DataFormatter, status: str, id: str):
        image_name = manager.workflow.metadata.image_name if manager.workflow else ""
        worker_name = context.get("worker_name") or "poly_gone"
        img_id = f"{status}_{id}_close"
        save_croped_image(image_name, img_id, polygon, worker_name)