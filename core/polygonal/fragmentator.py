# PerfectOCR/core/workflow/preprocessing/fragmentator.py
import cv2
import logging
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class PolygonFragmentator:
    
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.config = config
        fragmentator_params = self.config.get('fragmentation', {})
        self.min_area_factor = fragmentator_params.get('min_area_factor', 0.01)
        self.density_std_factor = fragmentator_params.get('density_std_factor', 1.0)
        self.approx_poly_epsilon = fragmentator_params.get('approx_poly_epsilon', 0.02)
        self.problematic_ids = set()

    def _intercept_polygons(self, cleaned_binarized_polygons: List[Dict], individual_polygons: List[Dict]) -> List[Dict[str, Any]]:
        """
        Primero, identifica polígonos problemáticos mediante un análisis estadístico de ladensidad de texto. Luego, fragmenta únicamente los polígonos problemáticos.
        Args:
            binarized_polygons (List[Dict]): Usados para el ANÁLISIS de densidad.
            individual_polygons (List[Dict]): Fuente de las imágenes para el RESULTADO FINAL.
        Returns:
            List[Dict[str, Any]]: Una lista unificada de polígonos originales y fragmentos nuevos.
        """
        if not individual_polygons or not cleaned_binarized_polygons:
            logger.warning("Se recibieron listas de polígonos vacías. No se puede procesar.")
            return individual_polygons

        # --- PASO 1: ANÁLISIS RÁPIDO - Medir densidad de texto en todos los polígonos ---
        poly_densities = []
        for poly in cleaned_binarized_polygons:
            poly_id = poly.get('polygon_id')
            bin_img = poly.get("binarized_img")
            if poly_id is None or bin_img is None or bin_img.size == 0:
                continue
            
            # Encontrar el contorno más grande para aproximarlo
            contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            
            main_contour = max(contours, key=cv2.contourArea)
            
            # Aproximar el polígono para que se ajuste mejor al texto
            perimeter = cv2.arcLength(main_contour, True)
            approximated_contour = cv2.approxPolyDP(main_contour, self.approx_poly_epsilon * perimeter, True)
            
            # Crear una máscara con el polígono ajustado
            mask = np.zeros_like(bin_img, dtype=np.uint8)
            cv2.drawContours(mask, [approximated_contour], -1, (255,), thickness=cv2.FILLED)
            
            # Calcular densidad dentro de la máscara
            mask_area = np.sum(mask == 255)
            if mask_area == 0:
                continue
                
            text_pixels_in_mask = np.sum(cv2.bitwise_and(bin_img, mask) == 255)
            density = text_pixels_in_mask / mask_area
            poly_densities.append({'id': poly_id, 'density': density})

        if not poly_densities:
            logger.warning("No se pudo calcular la densidad de ningún polígono.")
            return individual_polygons

        # --- PASO 2: VERIFICACIÓN ESTADÍSTICA - Identificar outliers ---
        densities = [d['density'] for d in poly_densities]
        mean_density = np.mean(densities)
        std_density = np.std(densities)
        
        # Usamos la media MENOS la desviación, ya que buscamos los que tienen menos "relleno"
        density_threshold = mean_density - self.density_std_factor * std_density
        problematic_ids = {d['id'] for d in poly_densities if d['density'] < density_threshold}
        self.problematic_ids = problematic_ids  # Guardar para acceso posterior

        logger.info(f"Análisis de densidad: media={mean_density:.3f}, std={std_density:.3f}. Se identificaron {len(problematic_ids)} polígonos problemáticos.")

        # --- PASO 3: CORRECCIÓN QUIRÚRGICA - Procesar y fragmentar solo los problemáticos ---
        temp_refined_polygons = []
        original_poly_map = {p['polygon_id']: p for p in individual_polygons}
        binarized_poly_map = {p['polygon_id']: p for p in cleaned_binarized_polygons}

        for poly_id, original_poly in sorted(original_poly_map.items(), key=lambda item: int(item[0].split('_')[-1])):
            if poly_id not in problematic_ids:
                # Polígono bueno, añadirlo a la lista final y marcar
                original_poly['was_fragmented'] = False
                temp_refined_polygons.append(original_poly)
                continue

            # --- Lógica de Fragmentación para polígonos problemáticos ---
            logger.info(f"Polígono problemático detectado ({poly_id}). Aplicando fragmentación.")
            binarized_polygon = binarized_poly_map.get(poly_id)
            if binarized_polygon is None or binarized_polygon.get("binarized_img") is None:
                original_poly['was_fragmented'] = False
                temp_refined_polygons.append(original_poly)
                continue
                
            bin_img_to_split = binarized_polygon["binarized_img"]
            
            internal_contours, _ = cv2.findContours(bin_img_to_split, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            poly_area = bin_img_to_split.shape[0] * bin_img_to_split.shape[1]
            adaptive_min_area = poly_area * self.min_area_factor
            valid_contours = [c for c in internal_contours if cv2.contourArea(c) > adaptive_min_area]
            
            if len(valid_contours) <= 1:
                original_poly['was_fragmented'] = False
                temp_refined_polygons.append(original_poly)
                continue

            sorted_contours = sorted(valid_contours, key=lambda c: cv2.boundingRect(c)[0])
            
            for i, contour_fragment in enumerate(sorted_contours):
                x, y, w, h = cv2.boundingRect(contour_fragment)
                fragment_img = original_poly['cropped_img'][y:y+h, x:x+w]
                if fragment_img.size == 0: continue

                # Heredar todas las claves del padre para mantener la consistencia
                new_poly_dict = original_poly.copy()
                
                # Actualizar las claves específicas del fragmento
                new_poly_dict['was_fragmented'] = True
                new_poly_dict['cropped_img'] = fragment_img
                
                # Calcular y actualizar la nueva geometría absoluta del fragmento
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
                temp_refined_polygons.append(new_poly_dict)

        # --- PASO 4: REASIGNACIÓN DE IDs ---
        refined_polygons = []
        for i, poly in enumerate(temp_refined_polygons):
            poly['polygon_id'] = f"poly_{i:04d}"
            refined_polygons.append(poly)

        logger.info(f"Análisis completado. Total de polígonos refinados: {len(refined_polygons)}")
        return refined_polygons
    
    def _get_problematic_ids(self) -> set:
        """Retorna los IDs de polígonos problemáticos del último procesamiento."""
        return self.problematic_ids