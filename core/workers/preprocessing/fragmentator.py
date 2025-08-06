# PerfectOCR/core/workflow/preprocessing/fragmentator.py
import cv2
import logging
import numpy as np
from typing import Dict, Any, Set

logger = logging.getLogger(__name__)

class PolygonFragmentator:
    
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.config = config
        fragmentator_params = self.config.get('fragmentation', {})
        self.min_area_factor = fragmentator_params.get('min_area_factor', 0.01)
        self.density_std_factor = fragmentator_params.get('density_std_factor', 1.0)
        self.approx_poly_epsilon = fragmentator_params.get('approx_poly_epsilon', 0.02)
        self.problematic_ids: Set[str] = set()

    def _intercept_polygons(self, binarized_polygons: Dict[str, np.ndarray], processing_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analiza polígonos binarizados para detectar problemas y fragmenta los polígonos originales correspondientes.
        
        Args:
            binarized_polygons: Diccionario polygon_id -> imagen binarizada (para ANÁLISIS)
            extracted_polygons: Diccionario polygon_id -> datos del polígono (para ACCIÓN)
            
        Returns:
            Dict[str, Any]: Diccionario de polígonos modificado con was_fragmented y perimeter
        """
        if not processing_dict or not binarized_polygons:
            logger.warning("Se recibieron diccionarios de polígonos vacíos. No se puede procesar.")
            return processing_dict if processing_dict is not None else {}

        # --- PASO 1: ANÁLISIS RÁPIDO - Medir perímetros de todos los contornos en cada polígono ---
        poly_perimeters = []
        for poly_id, bin_img in binarized_polygons.items():
            if bin_img is None or bin_img.size == 0:
                continue

            contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            # Calcula perímetro de todos los contornos
            perimeters = [cv2.arcLength(c, True) for c in contours]
            mean_perim = np.mean(perimeters)
            std_perim = np.std(perimeters)

            # Filtra contornos cuyo perímetro esté dentro de 2 desviaciones estándar del promedio
            valid_contours = [
                c for c, p in zip(contours, perimeters)
                if abs(p - mean_perim) <= 2 * std_perim
            ]

            num_valid = len(valid_contours)
            sum_valid_perim = sum([cv2.arcLength(c, True) for c in valid_contours])

            # Guarda el perímetro principal (el mayor) para polígonos no problemáticos
            main_perimeter = 0.0
            if valid_contours:
                main_perimeter = cv2.arcLength(max(valid_contours, key=cv2.contourArea), True)

            poly_perimeters.append({
                'id': poly_id,
                'num_valid_contours': num_valid,
                'sum_valid_perimeter': sum_valid_perim,
                'main_perimeter': main_perimeter
            })

        if not poly_perimeters:
            logger.warning("No se pudo calcular perímetros de ningún polígono.")
            return processing_dict

        # --- PASO 2: VERIFICACIÓN ESTADÍSTICA - Identificar outliers ---
        num_contours_list = [d['num_valid_contours'] for d in poly_perimeters]
        sum_perimeters_list = [d['sum_valid_perimeter'] for d in poly_perimeters]

        mean_num_contours = np.mean(num_contours_list)
        std_num_contours = np.std(num_contours_list)
        mean_sum_perim = np.mean(sum_perimeters_list)
        std_sum_perim = np.std(sum_perimeters_list)

        # Considera problemáticos los polígonos con muchos contornos (agrupamiento incorrecto) o perímetro total muy bajo (ruido)
        problematic_ids = {
            d['id'] for d in poly_perimeters
            if (d['num_valid_contours'] > mean_num_contours + self.density_std_factor * std_num_contours)
            or (d['sum_valid_perimeter'] < mean_sum_perim - self.density_std_factor * std_sum_perim)
        }
        self.problematic_ids = problematic_ids  # Guardar para acceso posterior

        logger.info(
            f"Análisis de contornos: media_n={mean_num_contours:.2f}, std_n={std_num_contours:.2f}, "
            f"media_perim={mean_sum_perim:.2f}, std_perim={std_sum_perim:.2f}. "
            f"Se identificaron {len(problematic_ids)} polígonos problemáticos."
        )

        refined_polygons = {}
        # Accede al diccionario interno 'polygons' antes de ordenar
        polygons_to_sort = processing_dict.get("polygons", {})
        sorted_polygons = sorted(polygons_to_sort.items(), key=lambda item: int(item[0].split('_')[-1]))
        current_position = 0
        
        for poly_id, original_poly in sorted_polygons:
            if poly_id not in problematic_ids:
                # Polígono bueno, añadirlo al diccionario final y marcar
                original_poly['was_fragmented'] = False
                # Agrega el perímetro principal (de la palabra)
                perim_entry = next((d for d in poly_perimeters if d['id'] == poly_id), None)
                if perim_entry:
                    original_poly['perimeter'] = perim_entry['main_perimeter']
                else:
                    original_poly['perimeter'] = 0.0
                
                # Asignar ID en la posición actual
                new_poly_id = f"poly_{current_position:04d}"
                refined_polygons[new_poly_id] = original_poly
                current_position += 1
                continue

            # --- Lógica de Fragmentación para polígonos problemáticos ---
            logger.info(f"Polígono problemático detectado ({poly_id}). Aplicando fragmentación.")
            bin_img_to_split = binarized_polygons.get(poly_id)
            if bin_img_to_split is None:
                original_poly['was_fragmented'] = False
                perim_entry = next((d for d in poly_perimeters if d['id'] == poly_id), None)
                if perim_entry:
                    original_poly['perimeter'] = perim_entry['main_perimeter']
                else:
                    original_poly['perimeter'] = 0.0
                
                new_poly_id = f"poly_{current_position:04d}"
                refined_polygons[new_poly_id] = original_poly
                current_position += 1
                continue

            internal_contours, _ = cv2.findContours(bin_img_to_split, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            poly_area = bin_img_to_split.shape[0] * bin_img_to_split.shape[1]
            adaptive_min_area = poly_area * self.min_area_factor
            valid_contours = [c for c in internal_contours if cv2.contourArea(c) > adaptive_min_area]

            if len(valid_contours) <= 1:
                original_poly['was_fragmented'] = False
                perim_entry = next((d for d in poly_perimeters if d['id'] == poly_id), None)
                if perim_entry:
                    original_poly['perimeter'] = perim_entry['main_perimeter']
                else:
                    original_poly['perimeter'] = 0.0
                
                new_poly_id = f"poly_{current_position:04d}"
                refined_polygons[new_poly_id] = original_poly
                current_position += 1
                continue

            # Fragmentar el polígono problemático
            sorted_contours = sorted(valid_contours, key=lambda c: cv2.boundingRect(c)[0])
            num_fragments = len(sorted_contours)
            
            for i, contour_fragment in enumerate(sorted_contours):
                x, y, w, h = cv2.boundingRect(contour_fragment)

                if 'cropped_img' not in original_poly or not isinstance(original_poly['cropped_img'], np.ndarray):
                    continue

                fragment_img = original_poly['cropped_img'][y:y+h, x:x+w]
                if fragment_img.size == 0:
                    continue

                new_poly_dict = original_poly.copy()

                new_poly_dict['was_fragmented'] = True
                new_poly_dict['cropped_img'] = fragment_img

                original_bbox = original_poly.get("geometry", {}).get("bounding_box", [0, 0, 0, 0])
                new_bbox = [
                    original_bbox[0] + x,
                    original_bbox[1] + y,
                    original_bbox[0] + x + w,
                    original_bbox[1] + y + h
                ]
                new_centroid = [new_bbox[0] + w / 2, new_bbox[1] + h / 2]

                new_poly_dict['geometry'] = {
                    'bounding_box': new_bbox,
                    'width': w,
                    'height': h,
                    'centroid': new_centroid
                }
                # Calcula el perímetro del fragmento (palabra)
                fragment_perimeter = cv2.arcLength(contour_fragment, True)
                new_poly_dict['perimeter'] = fragment_perimeter

                # Asignar ID en la posición actual
                new_poly_id = f"poly_{current_position:04d}"
                refined_polygons[new_poly_id] = new_poly_dict
                current_position += 1

        logger.info(f"Análisis completado. Total de polígonos refinados: {len(refined_polygons)}")
        return {"polygons": {
            poly_id: {
                "cropped_img": poly_data["cropped_img"],
                "was_fragmented": poly_data["was_fragmented"],
                "perimeter": poly_data.get("perimeter", 0.0)
            }
            for poly_id, poly_data in refined_polygons.items()
        }}
    
    def _get_problematic_ids(self) -> Set[str]:
        """Retorna los IDs de polígonos problemáticos del último procesamiento."""
        return self.problematic_ids
