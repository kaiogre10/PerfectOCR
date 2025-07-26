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

    def _intercept_polygons(self, binarized_poly: List[Dict], individual_polygons: List[Dict]) -> List[Dict]:
        """
        Primero, identifica polígonos problemáticos mediante un análisis estadístico de ladensidad de texto. Luego, fragmenta únicamente los polígonos problemáticos.
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
        
        # Usamos la media MENOS la desviación, ya que buscamos los que tienen menos "relleno"
        density_threshold = mean_density - self.density_std_factor * std_density
        problematic_ids = {d['id'] for d in poly_densities if d['density'] < density_threshold}
        self.problematic_ids = problematic_ids  # Guardar para acceso posterior

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
            binarized_polygon = binarized_poly_map.get(poly_id)
            if binarized_polygon is None or binarized_polygon.get("binarized_img") is None:
                refined_polygons.append(original_poly)
                continue
                
            bin_img_to_split = binarized_polygon["binarized_img"]
            
            # Encontrar contornos de los componentes internos
            internal_contours, _ = cv2.findContours(bin_img_to_split, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            poly_area = bin_img_to_split.shape[0] * bin_img_to_split.shape[1]
            adaptive_min_area = poly_area * self.min_area_factor
            valid_contours = [c for c in internal_contours if cv2.contourArea(c) > adaptive_min_area]
            
            if len(valid_contours) <= 1:
                refined_polygons.append(original_poly) # No se pudo fragmentar
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
    
    def _get_problematic_ids(self) -> set:
        """Retorna los IDs de polígonos problemáticos del último procesamiento."""
        return self.problematic_ids
    
    def es_poligono_problematico(bin_img: np.ndarray) -> bool:
        """
        Analiza una imagen binarizada para determinar si contiene múltiples palabras
        basándose en el espaciado entre componentes.

        Args:
            bin_img: La imagen del polígono, binarizada y limpia.

        Returns:
            True si se detectan espacios grandes (probablemente entre palabras), False en caso contrario.
        """
        # 1. Encuentra todos los componentes (caracteres)
        # stats contiene [x, y, width, height, area] para cada componente
        # el componente 0 es siempre el fondo, por lo que lo ignoramos.
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img, 4, cv2.CV_32S)

        # Si hay menos de 2 componentes (ej. una sola letra o nada), no puede ser problemático.
        if num_labels <= 2:
            return False

        # 2. Ordena los componentes de izquierda a derecha basándote en su coordenada X.
        # Ignoramos el componente 0 (fondo)
        componentes_ordenados = sorted(stats[1:], key=lambda s: s[0])

        # 3. Mide los espacios horizontales entre el final de un componente y el inicio del siguiente.
        gaps = []
        for i in range(len(componentes_ordenados) - 1):
            # Final del componente actual (x + width)
            fin_actual = componentes_ordenados[i][0] + componentes_ordenados[i][2]
            # Inicio del siguiente componente
            inicio_siguiente = componentes_ordenados[i+1][0]
            
            gap = inicio_siguiente - fin_actual
            if gap > 0: # Solo consideramos espacios positivos
                gaps.append(gap)

        if not gaps:
            return False # No hay espacios entre componentes

        # 4. Usa el método del Rango Intercuartílico (IQR) para encontrar outliers.
        # Un espacio anormalmente grande es un indicador de un espacio entre palabras.
        q1 = np.percentile(gaps, 25)
        q3 = np.percentile(gaps, 75)
        iqr = q3 - q1
        
        # El umbral para considerar un espacio como un outlier (anormalmente grande)
        # Este es un umbral estadístico estándar.
        umbral_outlier = q3 + 1.5 * iqr

        # Si CUALQUIER espacio es más grande que nuestro umbral, es un polígono problemático.
        if any(g > umbral_outlier for g in gaps):
            return True

        return False