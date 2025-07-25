# PerfectOCR/core/workflow/preprocessing/fragmentator.py
from sklearnex import patch_sklearn
patch_sklearn()
import cv2
import logging
import numpy as np
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class PolygonFragmentator:
    
    def __init__(self, config: Dict[str, Any], project_root: str):
        self.project_root = project_root
        self.fragmentator_config = config.get('polygon_config', {})
        
        fragmentator_params = self.fragmentator_config.get('fragmentation', {})
        self.min_area_factor = fragmentator_params.get('min_area_factor', 0.01)
        self.density_std_factor = fragmentator_params.get('density_std_factor', 1.0)
        self.approx_poly_epsilon = fragmentator_params.get('approx_poly_epsilon', 0.02)

    def _intercept_polygons(self, binarized_poly: List[Dict], individual_polygons: List[Dict]) -> List[Dict]:
        """
        Primero, identifica polígonos problemáticos mediante un análisis estadístico de la
        densidad de texto. Luego, fragmenta únicamente los polígonos problemáticos.
        Args:
            binarized_polygons (List[Dict]): Usados para el ANÁLISIS de densidad.
            individual_polygons (List[Dict]): Fuente de las imágenes para el RESULTADO FINAL.
        Returns:
            List[Dict]: Una lista unificada de polígonos originales y fragmentos nuevos.
        """
        if not individual_polygons or not binarized_poly:
            logger.warning("Se recibieron listas de polígonos vacías. No se puede procesar.")
            return individual_polygons

        # --- PASO 1: ANÁLISIS RÁPIDO - Medir densidad de texto en todos los polígonos ---
        poly_densities = []
        for poly in binarized_poly:
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
        
        # Un polígono es problemático si su densidad es muy baja (mucho espacio en blanco)
        # Usamos la media MENOS la desviación, ya que buscamos los que tienen menos "relleno"
        density_threshold = mean_density - self.density_std_factor * std_density
        problematic_ids = {d['id'] for d in poly_densities if d['density'] < density_threshold}
        
        logger.info(f"Análisis de densidad: media={mean_density:.3f}, std={std_density:.3f}. Se identificaron {len(problematic_ids)} polígonos problemáticos.")

        # --- PASO 3: CORRECCIÓN QUIRÚRGICA - Procesar y fragmentar solo los problemáticos ---
        refined_polygons = []
        original_poly_map = {p['polygon_id']: p for p in individual_polygons}
        binarized_poly_map = {p['polygon_id']: p for p in binarized_poly}

        for poly_id, original_poly in original_poly_map.items():
            if poly_id not in problematic_ids:
                # Polígono bueno, añadirlo a la lista final y continuar
                refined_polygons.append(original_poly)
                continue

            # --- Lógica de Fragmentación para polígonos problemáticos ---
            logger.info(f"Polígono problemático detectado ({poly_id}). Aplicando fragmentación.")
            binarized_poly: Optional[Dict] = binarized_poly_map.get(poly_id)
            if binarized_poly is None or binarized_poly.get("binarized_img") is None:
                refined_polygons.append(original_poly)
                continue
                
            bin_img_to_split = binarized_poly["binarized_img"]
            
            # Encontrar contornos de los componentes internos
            internal_contours, _ = cv2.findContours(bin_img_to_split, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # --- CAMBIO CLAVE: El área mínima ahora es adaptativa ---
            poly_area = bin_img_to_split.shape[0] * bin_img_to_split.shape[1]
            adaptive_min_area = poly_area * self.min_area_factor
            valid_contours = [c for c in internal_contours if cv2.contourArea(c) > adaptive_min_area]
            
            if len(valid_contours) <= 1:
                refined_polygons.append(original_poly) # No se pudo fragmentar, mantener original
                continue

            sorted_contours = sorted(valid_contours, key=lambda c: cv2.boundingRect(c)[0])
            
            for i, contour_fragment in enumerate(sorted_contours):
                x, y, w, h = cv2.boundingRect(contour_fragment)
                fragment_img = original_poly['cropped_img'][y:y+h, x:x+w]
                
                if fragment_img.size == 0: continue

                new_poly_id = f"{poly_id}_frag{i+1}"
                new_polygon = {
                    "polygon_id": new_poly_id, "cropped_img": fragment_img,
                    "x_min": original_poly['x_min'] + x, "y_min": original_poly['y_min'] + y,
                    "width": w, "height": h, "parent_id": poly_id,
                    **{k: v for k, v in original_poly.items() if k not in 
                       ['polygon_id', 'cropped_img', 'x_min', 'y_min', 'width', 'height']}
                }
                refined_polygons.append(new_polygon)

        logger.info(f"Análisis completado. Total de polígonos refinados: {len(refined_polygons)}")
        return refined_polygons

