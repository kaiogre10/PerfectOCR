# PerfectOCR/core/workflow/preprocessing/poly_gone.py
import cv2
import numpy as np
from typing import Tuple, List, Dict, Any


class PolygonExtractor:
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.config = config

    def extract_all_polygons(self, deskewed_img: np.ndarray, all_polygons_coords: List[List[List[float]]]) -> List[np.ndarray]:
        """
        Extrae cada polígono de la lista de coordenadas y devuelve una lista de imágenes recortadas.
        """
        if not all_polygons_coords:
            return []

        cropped_images = []
        padding = 5  # Padding para evitar cortar bordes

        for single_polygon_coords in all_polygons_coords:
            if not single_polygon_coords or len(single_polygon_coords) < 3:
                continue

            try:
                arr = np.array(single_polygon_coords, dtype=np.int32)
                rect = cv2.boundingRect(arr)
                x, y, w, h = rect

                # Aplicar padding
                x_pad, y_pad = max(0, x - padding), max(0, y - padding)
                w_pad, h_pad = w + (2 * padding), h + (2 * padding)

                # Asegurar que el recorte no se salga de los límites de la imagen
                img_h, img_w = deskewed_img.shape[:2]
                x_end = min(x_pad + w_pad, img_w)
                y_end = min(y_pad + h_pad, img_h)

                # Recortar y añadir a la lista
                polygon_img = deskewed_img[y_pad:y_end, x_pad:x_end]
                if polygon_img.size > 0:
                    cropped_images.append(polygon_img)
            except Exception as e:
                print(f"Error recortando polígono: {e}")
                continue

        return cropped_images

    def _extract_polygon_image(self, deskewed_img: np.ndarray, polygons_coords: List[List[float]], metadata: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int, int, int], Tuple[int, int]]:
        """Extrae la imagen de un único polígono y devuelve su ROI (Region of Interest) con un pequeño padding."""
        # Esta función ahora puede considerarse obsoleta o para uso de un solo polígono.
        # Por simplicidad, la dejaremos como está pero no la usaremos en el flujo principal.
        if not polygons_coords or len(polygons_coords) == 0:
            raise ValueError("polygons_coords está vacío o None")
        
        all_points = []
        for poly in polygons_coords:
            all_points.extend(poly)

        if not all_points:
             raise ValueError("polygons_coords no contiene puntos válidos")
        
        arr = np.array(all_points, dtype=np.int32)

        rect = cv2.boundingRect(arr)
        x, y, w, h = rect

        # Padding para evitar cortar bordes al procesar
        padding = 5
        x_pad, y_pad = max(0, x - padding), max(0, y - padding)
        w_pad, h_pad = w + (2 * padding), h + (2 * padding)

        # Asegurar que el polígono no se salga de los límites de la imagen
        img_h, img_w = deskewed_img.shape[:2]
        x_end = min(x_pad + w_pad, img_w)
        y_end = min(y_pad + h_pad, img_h)

        polygon_img = deskewed_img[y_pad:y_end, x_pad:x_end]

        return polygon_img, (x_pad, y_pad, x_end - x_pad, y_end - y_pad), metadata