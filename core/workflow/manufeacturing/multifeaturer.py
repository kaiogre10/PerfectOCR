# PerfectOCR/core/workflow/preprocessing/manufacturer.py
from sklearnex import patch_sklearn
patch_sklearn
from skimage.measure import label, regionprops
import cv2
from typing import List, Dict 
import numpy as np
        
class MultiFeacturer:
    def _extract_region_features(self, binary_image: np.ndarray) -> List[Dict]:
        # Etiquetar regiones conectadas
        labeled_img = label(binary_image)
        regions = regionprops(labeled_img, intensity_image=binary_image)
        
        features_list = []
        img_area = binary_image.shape[0] * binary_image.shape[1]
        
        for region in regions:
                
            features = {
                # Propiedades básicas de área y forma
                'area': region.area,  # Número de píxeles en la región
                'bbox': region.bbox,  # Bounding box (min_row, min_col, max_row, max_col)
                'bbox_area': region.bbox_area,  # Área del bounding box
                'convex_area': region.convex_area,  # Área de la envolvente convexa
                'filled_area': region.filled_area,  # Área con huecos rellenados
                
                # Centroides y coordenadas
                'centroid': region.centroid,  # Centro de masa (row, col)
                'local_centroid': region.local_centroid,  # Centroide relativo al bbox
                'weighted_centroid': region.weighted_centroid,  # Centroide ponderado por intensidad
                'weighted_local_centroid': region.weighted_local_centroid,  # Centroide ponderado local
                'coords': region.coords,  # Coordenadas de todos los píxeles de la región
                
                # Propiedades geométricas
                'eccentricity': region.eccentricity,  # Excentricidad de la elipse (0=círculo, 1=línea)
                'extent': region.extent,  # Proporción area/bbox_area
                'major_axis_length': region.major_axis_length,  # Longitud del eje mayor
                'minor_axis_length': region.minor_axis_length,  # Longitud del eje menor
                'orientation': region.orientation,  # Ángulo del eje mayor respecto al horizontal
                'equivalent_diameter': region.equivalent_diameter,  # Diámetro de círculo con misma área
                'perimeter': region.perimeter,  # Perímetro de la región
                'perimeter_crofton': region.perimeter_crofton,  # Perímetro usando fórmula de Crofton
                'solidity': region.solidity,  # Proporción area/convex_area (solidez)
                'equivalent_diameter_area': region.equivalent_diameter_area, # Diámetro de un círculo con la misma área que la región
                'feret_diameter_max': region.feret_diameter_max, # Diámetro máximo de Feret

                # Propiedades topológicas
                'euler_number': region.euler_number,  # Número Euler (objetos - huecos)
                
                # Propiedades de intensidad
                'max_intensity': region.max_intensity,  # Valor máximo de intensidad
                'mean_intensity': region.mean_intensity,  # Valor promedio de intensidad
                'min_intensity': region.min_intensity,  # Valor mínimo de intensidad
                
                # Momentos estadísticos
                'moments': region.moments,  # Momentos espaciales
                'moments_central': region.moments_central,  # Momentos centrales
                'moments_normalized': region.moments_normalized,  # Momentos normalizados
                'moments_hu': region.moments_hu,  # Momentos de Hu (invariantes)
                'weighted_moments': region.weighted_moments,  # Momentos ponderados por intensidad
                'weighted_moments_central': region.weighted_moments_central,  # Momentos centrales ponderados
                'weighted_moments_normalized': region.weighted_moments_normalized,  # Momentos normalizados ponderados
                'weighted_moments_hu': region.weighted_moments_hu,  # Momentos Hu ponderados
                
                # Tensor de inercia
                'inertia_tensor': region.inertia_tensor,  # Tensor de inercia 2x2
                'inertia_tensor_eigvals': region.inertia_tensor_eigvals,  # Eigenvalores del tensor
                
                # Imágenes y metadatos
                'image': region.image,  # Imagen binaria de la región
                'convex_image': region.convex_image,  # Imagen de la envolvente convexa
                'filled_image': region.filled_image,  # Imagen con huecos rellenados
                'intensity_image': region.intensity_image,  # Imagen de intensidad de la región
                'label': region.label,  # Etiqueta de la región
                'slice': region.slice,  # Slice para extraer la región de la imagen original
            }
            features_list.append(features)
        
        return features_list

