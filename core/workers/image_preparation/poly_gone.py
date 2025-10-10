# PerfectOCR/core/preprocessing/poly_gone.py
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from core.factory.abstract_worker import ImagePrepAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons

logger = logging.getLogger(__name__)

class PolygonExtractor(ImagePrepAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.config = config
        self.worker_config = config.get('polygon_extractor', {})
        self.enabled_outputs = self.config.get("enabled_outputs", {})
        self.output = self.enabled_outputs.get("cropped_img", False)

    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """Extrae polígonos en batch usando operaciones vectorizadas para optimizar el recorte.
        Siguiendo el patrón: Análisis → Decisión Vectorizada → Aplicación"""
        try:
            import time
            start_time = time.time()
            
            full_img = manager.get_full_img()
            if full_img is None:
                logger.error(f"No Hay full_img en el Formatter")
                return False
                
            logger.debug("Full_img obtenida con éxito")
                
            polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
            img_dims: Dict[str, int] = {}
            if manager.workflow and hasattr(manager.workflow, "metadata") and hasattr(manager.workflow.metadata, "img_dims"):
                img_dims = dict(getattr(manager.workflow.metadata, "img_dims", {}))
                
            img_h = img_dims.get("height") or 0
            
            img_w = img_dims.get("width") or 0
                        
            if not polygons:
                logger.warning("PolygonExtractor: No se encontraron polígonos para procesar.")
                return True

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
                return True

            # 2. Fase de Decisión Vectorizada: Calcular todos los recortes con padding
            padding = self.worker_config.get("cropping_padding")
            
            # Convertir a array NumPy para operaciones vectorizadas
            bboxes_array = np.array(all_bboxes)  # shape: (n_polygons, 4)
            
            # Calcular coordenadas con padding usando operaciones vectorizadas
            x1, y1, x2, y2 = bboxes_array[:, 0], bboxes_array[:, 1], bboxes_array[:, 2], bboxes_array[:, 3]
            
            # Aplicar padding y clipping de una vez
            px1 = np.maximum(0, x1 - padding)
            py1 = np.maximum(0, y1 - padding)
            px2 = np.minimum(img_w, x2 + padding)
            py2 = np.minimum(img_h, y2 + padding)
            
            # Validar dimensiones usando operaciones vectorizadas
            valid_dims: np.ndarray[Any, Any] = (px2 > px1) & (py2 > py1)
            valid_indices = np.where(valid_dims)[0]
            
            if len(valid_indices) == 0:
                logger.warning("PolygonExtractor: No hay recortes válidos después del padding.")
                return True

            # Calcular centroides con padding de forma vectorizada
            padded_centroids_x = (px1[valid_indices] + px2[valid_indices]) / 2
            padded_centroids_y = (py1[valid_indices] + py2[valid_indices]) / 2
            
            # 3. Fase de Aplicación: Extraer imágenes solo para índices válidos
            cropped_images: Dict[str, np.ndarray[Any, np.dtype[np.uint8]]] = {}
            cropped_geometries: Dict[str, Dict[str, Any]] = {}
            discarded_poly_ids: List[str] = []
            valid_poly_ids: List[str] = []

            bin_interval: Tuple[int, int] = self.worker_config.get("bin_interval", [10, 245])

            for i, idx in enumerate(valid_indices):
                poly_id: str = poly_ids_order[idx] #type: ignore
                crop_x1, crop_y1 = int(px1[idx]), int(py1[idx])
                crop_x2, crop_y2 = int(px2[idx]), int(py2[idx])
                cropped: np.ndarray[Any, np.dtype[np.uint8]] = full_img[crop_y1:crop_y2, crop_x1:crop_x2].copy()

                if cropped.size == 0:
                    discarded_poly_ids.append(f"{poly_id} (vacío)")
                    continue

                img_mean = cropped.mean()
                if img_mean < bin_interval[0] or img_mean > bin_interval[1]:
                    discarded_poly_ids.append(f"{poly_id} (blanco/negro)")
                    continue

                # Solo los válidos se guardan
                new_id = f"poly_{len(valid_poly_ids):04d}"
                cropped_images[new_id] = cropped
                valid_poly_ids.append(poly_id)

                poly_height, poly_width = cropped.shape[:2]
                cropped_geometries[new_id] = {
                    "padd_centroid": [(padded_centroids_x[i]), (padded_centroids_y[i])],
                    "padding_coords": [(crop_x1), (crop_y1), (crop_x2), (crop_y2)],
                    "croppy_dims": {
                        "poly_height": poly_height, 
                        "poly_width":  poly_width,
                    }
                }

            # Eliminar los descartados del manager.workflow.polygons
            for poly_id in discarded_poly_ids:
                pid = poly_id.split(" ")[0]
                if pid in manager.workflow.polygons:
                    del manager.workflow.polygons[pid]

            # Reindexar los polígonos válidos en el manager
            import dataclasses
            new_polygons = {}
            for idx, old_id in enumerate(valid_poly_ids):
                new_id = f"poly_{idx:04d}"
                poly_obj = manager.workflow.polygons[old_id]
                poly_obj = dataclasses.replace(poly_obj, polygon_id=new_id)
                new_polygons[new_id] = poly_obj
            manager.workflow.polygons = new_polygons

            if discarded_poly_ids:
                logger.warning(f"PolygonExtractor: Se eliminaron {len(discarded_poly_ids)} polígonos no válidos: {', '.join(discarded_poly_ids)}")
            
            # Guardar resultados
            success = manager.save_cropped_images(cropped_images, cropped_geometries)
            if not success:
                logger.error("PolygonExtractor: Error al guardar imágenes recortadas en el workflow")
                return False

            if self.output:
                image_name = manager.workflow.metadata.image_name if manager.workflow else ""
                self._save_debug_image(context, cropped_images, image_name)

            # Liberamos la imagen del contexto y del workflow para ahorrar memoria
            manager.update_full_img(None)
            
            total_time = time.time() - start_time
            extracted_count = len(cropped_images)
            logger.debug(f"PolygonExtractor batch completado: {extracted_count} recortes en {total_time:.3f}s. 'full_img' liberada.")
            
            return True

        except Exception as e:
            logger.error(f"Error en PolygonExtractor: {e}", exc_info=True)
            return False

    def _save_debug_image(self, context: Dict[str, Any], cropped_images: Dict[str, np.ndarray[Any, np.dtype[np.uint8]]], image_name: str):
        from services.output_service import save_image
        import os

        output_paths = context.get("output_paths", [])
        
        if not output_paths:
            logger.error("No se especificaron rutas de salida para guardar imágenes de debug de poly_gone.")
            return

        for poly_id, cropped in cropped_images.items():
            for path in output_paths:
                output_dir = os.path.join(path, "poly_gone")
                file_name = f"{image_name}_{poly_id}_cropped_img.png"
                save_image(cropped, output_dir, file_name)
            logger.info(f"Imagen de debug de poly_gone para '{poly_id}' guardada en {len(output_paths)} ubicaciones.")