# PerfectOCR/core/preprocessing/poly_gone.py
import numpy as np
import logging
from typing import Dict, Any, List
from core.factory.abstract_worker import ImagePrepAbstractWorker
from core.domain.data_formatter import DataFormatter

logger = logging.getLogger(__name__)

class PolygonExtractor(ImagePrepAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.config = config

    def process(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        try:
            full_img = context.get("full_img")
            if full_img is None:
                logger.error("PolygonExtractor: 'full_img' no encontrado en el contexto.")
                return False

            dict_data = manager.get_dict_data()
            polygons:  Dict[str, Any] = dict_data.get("image_data", {}).get("polygons", {})
            if not polygons:
                logger.warning("PolygonExtractor: No se encontraron polígonos para procesar.")
                return True

            padding = int(self.config.get("cropping_padding", 5))
            img_h, img_w = full_img.shape[:2]

            # Asignar line_id usando la lógica del lineal_reconstructor
            id_map = self.assign_line_id(polygons)

            # Diccionario para almacenar las imágenes recortadas
            cropped_images: Dict[str, np.ndarray[Any, Any]] = {}
            cropped_geometries: Dict[str, Dict[str, Any]] = {}
            extracted_count = 0
            
            for poly_id, poly_data in polygons.items():
                # Usar geometry.bounding_box directamente
                bbox: List[float] = poly_data.get("geometry", {}).get("bounding_box")
                if not bbox or len(bbox) != 4:
                    logger.warning(f"PolygonExtractor: Bounding box inválido para {poly_id}")
                    continue
                
                x1, y1, x2, y2 = map(int, bbox)
                px1, py1 = max(0, x1 - padding), max(0, y1 - padding)
                px2, py2 = min(img_w, x2 + padding), min(img_h, y2 + padding)
                if px2 <= px1 or py2 <= py1:
                    logger.warning(f"PolygonExtractor: Dimensiones inválidas para {poly_id}")
                    continue

                cropped = full_img[py1:py2, px1:px2].copy()
                if cropped.size == 0:
                    logger.warning(f"PolygonExtractor: Imagen recortada vacía para {poly_id}")
                    continue

                # Guardamos la imagen recortada en nuestro diccionario temporal
                
                                
                cropped_images[poly_id] = cropped

                # Guardar la geometría del recorte (bounding box ajustada) como dict
                cropped_geometries[poly_id] = {
                    "padding_bbox": [px1, py1, px2, py2],
                    "padd_centroid": [(px1 + px2) / 2, (py1 + py2) / 2],
                    "padding_coords": [px1, py1, px2, py2]
                }
                extracted_count += 1

            # Guardamos todas las imágenes recortadas en el workflow
            
            success = manager.save_cropped_images(cropped_images, id_map, cropped_geometries)
            if not success:
                logger.error("PolygonExtractor: Error al guardar imágenes recortadas en el workflow")
                return False

            # Liberamos la imagen del contexto y del workflow para ahorrar memoria
            context["full_img"] = None
            manager.update_full_img(None)
            
            logger.info(f"PolygonExtractor: {extracted_count} recortes creados. 'full_img' liberada.")
            return True

        except Exception as e:
            logger.error(f"Error en PolygonExtractor: {e}", exc_info=True)
            return False

    def assign_line_id(self, polygons: Dict[str, Any]) -> Dict[str, str]:
        """
        Asigna un 'line_id' a cada polígono.
        Devuelve un mapeo polygon_id -> line_id.
        """
        if not polygons:
            logger.warning("No hay polígonos para asignar line_id.")
            return {}

        # Ordenar por la coordenada Y del centroide para simular líneas
        prepared_sorted = sorted(
            polygons.values(),
            key=lambda p: p.get("geometry", {}).get("centroid", [0, 0])[1]
        )
        id_map: Dict[str, str] = {}
        line_counter = 1
        last_centroid_y = None
        line_id = f"line_{line_counter:04d}"

        for poly in prepared_sorted:
            centroid = poly.get("geometry", {}).get("centroid")
            if centroid is None:
                continue
            centroid_y = centroid[1]
            if last_centroid_y is not None and abs(centroid_y - last_centroid_y) > 20:
                # Si la diferencia en Y es significativa, nueva línea
                line_counter += 1
                line_id = f"line_{line_counter:04d}"
            poly_id = poly.get("polygon_id")
            if poly_id:
                id_map[poly_id] = line_id
            last_centroid_y = centroid_y

        return id_map