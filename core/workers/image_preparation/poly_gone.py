# PerfectOCR/core/preprocessing/poly_gone.py
import numpy as np
import logging
from typing import Dict, Any, List
from core.factory.abstract_worker import ImagePrepAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons

logger = logging.getLogger(__name__)

class PolygonExtractor(ImagePrepAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = self.config.get('cutting', {})
        self.enabled_outputs = self.config.get("enabled_outputs", {})
        self.output = self.enabled_outputs.get("cropped_img", False)

    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """Extrae polígonos en batch usando operaciones vectorizadas para optimizar el recorte.
        Siguiendo el patrón: Análisis → Decisión Vectorizada → Aplicación"""
        try:
            import time
            start_time = time.time()
            
            img_obj = manager.get_full_img()
            full_img = img_obj.full_img if img_obj is not None else None
            if full_img is None:
                logger.error(f"No Hay full_img en el Formatter")
                return False
            logger.debug("Full_img obtenida con éxito")
                
            polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
            img_dims: Dict[str, int] = {}
            if manager.workflow and hasattr(manager.workflow, "metadata") and hasattr(manager.workflow.metadata, "img_dims"):
                img_dims = dict(getattr(manager.workflow.metadata, "img_dims", {}))
                
            img_h = img_dims.get("height")
            
            img_w = img_dims.get("width") 
                        
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
            padding = self.config.get("cropping_padding", 5)
            
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
            valid_dims = (px2 > px1) & (py2 > py1)
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
            
            for i, idx in enumerate(valid_indices):
                poly_id: str = poly_ids_order[idx] # type: ignore
                
                # Coordenadas calculadas vectorialmente
                crop_x1, crop_y1 = int(px1[idx]), int(py1[idx])
                crop_x2, crop_y2 = int(px2[idx]), int(py2[idx])
                
                # Extraer imagen
                cropped: np.ndarray[Any, np.dtype[np.uint8]] = full_img[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                
                if cropped.size == 0:
                    logger.warning(f"PolygonExtractor: Imagen recortada vacía para {poly_id}")
                    continue
                
                cropped_images[poly_id] = cropped
                
                poly_height, poly_width = cropped.shape[:2]
                
                # Guardar geometría usando resultados vectorizados
                cropped_geometries[poly_id] = {
                    "padd_centroid": [float(padded_centroids_x[i]), float(padded_centroids_y[i])],
                    "padding_coords": [int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)],
                    "croppy_dims": {
                        "poly_height": poly_height, 
                        "poly_width":  poly_width,
                    }
                }

            # Guardar resultados
            success = manager.save_cropped_images(cropped_images, cropped_geometries)
            if not success:
                logger.error("PolygonExtractor: Error al guardar imágenes recortadas en el workflow")
                return False

            if self.output:
                self._save_debug_image(context, cropped_images)

            # Liberamos la imagen del contexto y del workflow para ahorrar memoria
            context["full_img"] = None
            manager.update_full_img(None)
            
            total_time = time.time() - start_time
            extracted_count = len(cropped_images)
            logger.debug(f"PolygonExtractor batch completado: {extracted_count} recortes en {total_time:.3f}s. 'full_img' liberada.")
            
            return True

        except Exception as e:
            logger.error(f"Error en PolygonExtractor: {e}", exc_info=True)
            return False

    def _save_debug_image(self, context: Dict[str, Any], cropped_images: Dict[str, np.ndarray[Any, np.dtype[np.uint8]]]):
        from services.output_service import save_image
        import os

        output_paths = context.get("output_paths", [])
        if not output_paths:
            logger.debug("No se especificaron rutas de salida para guardar imágenes de debug de poly_gone.")
            return

        for poly_id, cropped in cropped_images.items():
            for path in output_paths:
                output_dir = os.path.join(path, "poly_gone")
                file_name = f"{poly_id}_poly_gone.png"
                save_image(cropped, output_dir, file_name)
            logger.debug(f"Imagen de debug de poly_gone para '{poly_id}' guardada en {len(output_paths)} ubicaciones.")