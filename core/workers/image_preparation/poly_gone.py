# core/preprocessing/poly_gone.py
import numpy as np
import logging
import time
import math
import dataclasses
from typing import Dict, Any, List
from core.factory.abstract_worker import ImagePrepAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons
from core.utils.image_utils import calculate_img_values

logger = logging.getLogger(__name__)

class PolygonExtractor(ImagePrepAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get('polygon_extractor', {})
        self.angle_thr = self.worker_config["angle_thr"]
        self.bin_interval = config["bin_interval"]
        self.padding = self.worker_config.get("cropping_padding")
        self.output = config.get("cropped_img", False)
        self.filtered_ouputs = config.get("final_polys", False)
        self.disoutput = config.get("discarded_polys", False)

    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """Extrae polígonos en batch usando operaciones vectorizadas para optimizar el recorte."""
        start_time = time.time()
        try:
            img_obj = manager.get_full_img()
            full_img = img_obj.full_img if img_obj is not None else None
            if full_img is None:
                logger.error(f"No Hay full_img en el Formatter")
                return False
                
            image_name = manager.workflow.metadata.image_name if manager.workflow else ""
            img_h = full_img.shape[0]
            img_w = full_img.shape[1]

            logger.debug("Full_img obtenida con éxito")
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
                    
                all_bboxes.append(bbox)
                poly_ids_order.append(poly_id)

            if not all_bboxes:
                logger.warning("PolygonExtractor: No hay bboxes válidos para procesar.")
                return False

            # 2. Fase de Decisión Vectorizada: Calcular todos los recortes con self.padding
            bboxes_array = np.array(all_bboxes)  # shape: (n_polygons, 4)
            
            # Calcular coordenadas con self.padding usando operaciones vectorizadas
            x1, y1, x2, y2 = bboxes_array[:, 0], bboxes_array[:, 1], bboxes_array[:, 2], bboxes_array[:, 3]
            
            # Aplicar self.padding y clipping de una vez
            px1 = np.maximum(0, x1 - self.padding)
            py1 = np.maximum(0, y1 - self.padding)
            px2 = np.minimum(img_w, x2 + self.padding)
            py2 = np.minimum(img_h, y2 + self.padding)
            
            # Liberar la imagen completa lo antes posible
            if manager.update_full_img(corrected=False, full_img=None):
                logger.info(f"full_img: '{image_name}' liberada")
            
            # Validar dimensiones usando operaciones vectorizadas
            valid_dims: np.ndarray[Any, Any] = (px2 > px1) & (py2 > py1)
            valid_indices = np.where(valid_dims)[0]
            
            if len(valid_indices) == 0:
                logger.warning("PolygonExtractor: No hay recortes válidos después del padding.")
                return False

            # Calcular centroides con padding de forma vectorizada
            padded_centroids_x = (px1[valid_indices] + px2[valid_indices]) / 2
            padded_centroids_y = (py1[valid_indices] + py2[valid_indices]) / 2
            
            # 3. Fase de Aplicación: Extraer imágenes solo para índices válidos
            cropped_images: Dict[str, np.ndarray[Any, np.dtype[np.uint8]]] = {}
            cropped_geometries: Dict[str, Dict[str, Any]] = {}
            discarded_poly_ids: List[str] = []
            valid_poly_ids: List[str] = []

            poly_data_to_filter: List[Dict[str, Any]] = []
            for i, idx in enumerate(valid_indices):
                poly_id: str = poly_ids_order[idx] # type: ignore
                crop_x1, crop_y1 = int(px1[idx]), int(py1[idx])
                crop_x2, crop_y2 = int(px2[idx]), int(py2[idx])
                cropped: np.ndarray[Any, np.dtype[np.uint8]] = full_img[crop_y1:crop_y2, crop_x1:crop_x2].copy()

                if self.output:
                    from services.output_service import save_croped_image
                    worker_name = context.get("worker_name") or "poly_gone"
                    output_paths = context["output_paths"]
                    pid = f"{image_name}_{poly_id}_all_{worker_name}"
                    save_croped_image(image_name, pid, cropped, output_paths, worker_name) # type: ignore
                
                poly_mean, dims = calculate_img_values(cropped)
                bbox_width = dims[1]
                bbox_height = dims[0]
                angle = math.degrees(math.atan2(bbox_height, bbox_width))
                
                # logger.info(f"{poly_id} AREA: {cropped.size}")
                poly_data_to_filter.append({
                    "poly_id": poly_id,
                    "cropped": cropped,
                    "angle": angle,
                    "i": i,
                    "coords": (crop_x1, crop_y1, crop_x2, crop_y2),
                    "poly_mean": poly_mean
                })

            if not poly_data_to_filter:
                logger.warning("PolygonExtractor: No hay polígonos válidos para procesar.")
                return False 

            valid_polygons_data: List[Dict[str, Any]] = []
            for p_data in poly_data_to_filter:

                if p_data['angle'] < self.angle_thr[1] and self.angle_thr[0] < p_data['angle']:
                    discarded_poly_ids.append(f"{p_data['poly_id']}, {p_data['angle']}")
                    # logger.info(f"ELIMINADO '{p_data['poly_id']}': Angulo = {p_data['angle']}°")

                elif p_data['poly_mean'] < self.bin_interval[0] or p_data['poly_mean'] > self.bin_interval[1]:
                    discarded_poly_ids.append(f"{p_data['poly_id']}, {p_data['poly_mean']}")
                    logger.info(f"ELIMINADO '{p_data['poly_id']}': FUERA DE RANGO = {p_data['poly_mean']}")
                    
                if self.disoutput:
                    from services.output_service import save_croped_image
                    worker_name = context.get("worker_name") or "poly_gone"
                    output_paths = context["output_paths"]
                    pid = f"{image_name}_{p_data['poly_id']}_deleted_{worker_name}"
                    save_croped_image(image_name, pid, p_data['cropped'], output_paths, worker_name)

                else:
                    valid_polygons_data.append(p_data)
                    valid_poly_ids.append(p_data['poly_id'])

            for i, p_data in enumerate(valid_polygons_data):
                new_id = f"poly_{i:04d}"
                cropped_images[new_id] = p_data["cropped"]

                poly_height, poly_width = p_data["cropped"].shape[:2]
                centroid_idx = p_data["i"]
                crop_x1, crop_y1, crop_x2, crop_y2 = p_data["coords"]

                cropped_geometries[new_id] = {
                    "padd_centroid": [(padded_centroids_x[centroid_idx]), (padded_centroids_y[centroid_idx])],
                    "padding_coords": [crop_x1, crop_y1, crop_x2, crop_y2],
                    "croppy_dims": {
                        "poly_height": poly_height,
                        "poly_width": poly_width,
                    }
                }

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
                poly_obj = manager.workflow.polygons[old_id] # type: ignore
                poly_obj = dataclasses.replace(poly_obj, polygon_id=new_id)
                new_polygons[new_id] = poly_obj
                    
            manager.workflow.polygons = new_polygons
            
            if not manager.save_cropped_images(cropped_images, cropped_geometries):
                logger.error("No se pudieron guardar las imagenes en el manager")
                return False
            
            total_time = time.time() - start_time
                
            extracted_count = len(cropped_images)
            logger.debug(f"'{extracted_count}' polígonos recortados en {total_time:.6f}s.")

            if self.filtered_ouputs:
                from services.output_service import save_croped_image
                worker_name = context.get("worker_name") or "poly_gone"
                output_paths = context["output_paths"]
                polygons = manager.workflow.polygons if manager.workflow else {}

                for poly_id, polygon in polygons.items():
                    pid = f"{image_name}_{poly_id}_filtered_{worker_name}"
                    cropped_img = polygon.cropped_img.cropped_img if polygon.cropped_img else None
                    save_croped_image(image_name, pid, cropped_img, output_paths, worker_name) #type: ignore
            
            return True

        except Exception as e:
            logger.error(f"Error en PolygonExtractor: {e}", exc_info=True)
            return False
        