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

    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        """Extrae polígonos en batch usando operaciones vectorizadas para optimizar el recorte.
        Siguiendo el patrón: Análisis → Decisión Vectorizada → Aplicación"""
        try:
            import time
            start_time = time.time()
            
            full_img = context.get("full_img")
            if full_img is None:
                logger.warning("PolygonExtractor: 'full_img' no encontrado en el contexto.")
                return False
                
            polygons: Dict[str, Polygons] = manager.get_polygons()
            img_h: int = context.get("metadata", {}).get("img_dims", {}).get("height")
            img_w: int = context.get("metadata", {}).get("img_dims", {}).get("width")
            
            if not polygons:
                logger.warning("PolygonExtractor: No se encontraron polígonos para procesar.")
                return True

            # 1. Fase de Análisis: Recopilar todas las bounding boxes
            poly_ids_order: List[str] = []
            all_bboxes: List[np.ndarray[Any, Any]] = []
            
            for poly_id, polygon in polygons.items():
                # Acceso directo a arrays NumPy desde la dataclass
                bbox = polygon.geometry.bounding_box  # Ya es np.ndarray
                if bbox.size != 4:
                    logger.warning(f"PolygonExtractor: Bounding box inválido para {poly_id}")
                    continue
                    
                all_bboxes.append(bbox)
                poly_ids_order.append(poly_id)

            if not all_bboxes:
                logger.warning("PolygonExtractor: No hay bboxes válidos para procesar.")
                return True

            # 2. Fase de Decisión Vectorizada: Calcular todos los recortes con padding
            padding = int(self.config.get("cropping_padding", 5))
            
            # Convertir a array NumPy para operaciones vectorizadas
            bboxes_array = np.array(all_bboxes, dtype=np.int32)  # shape: (n_polygons, 4)
            
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
                poly_id = poly_ids_order[idx]
                
                # Coordenadas calculadas vectorialmente
                crop_x1, crop_y1 = px1[idx], py1[idx]
                crop_x2, crop_y2 = px2[idx], py2[idx]
                
                # Extraer imagen
                cropped = full_img[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                
                if cropped.size == 0:
                    logger.warning(f"PolygonExtractor: Imagen recortada vacía para {poly_id}")
                    continue
                
                cropped_images[poly_id] = cropped
                
                # Guardar geometría usando resultados vectorizados
                cropped_geometries[poly_id] = {
                    "padd_centroid": [float(padded_centroids_x[i]), float(padded_centroids_y[i])],
                    "padding_coords": [int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)],
                }

            # Guardar resultados
            success = manager.save_cropped_images(cropped_images, cropped_geometries)
            if not success:
                logger.error("PolygonExtractor: Error al guardar imágenes recortadas en el workflow")
                return False

            # Liberamos la imagen del contexto y del workflow para ahorrar memoria
            context["full_img"] = None
            manager.update_full_img(None)
            
            total_time = time.time() - start_time
            extracted_count = len(cropped_images)
            logger.info(f"PolygonExtractor batch completado: {extracted_count} recortes en {total_time:.3f}s. 'full_img' liberada.")
            
            return True

        except Exception as e:
            logger.error(f"Error en PolygonExtractor: {e}", exc_info=True)
            return False