# PerfectOCR/core/workflow/preprocessing/manufacturer.py
from sklearnex import patch_sklearn
patch_sklearn
from skimage.measure import label, regionprops
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
                'area': region.area,
                'centroid': region.centroid,
                'bbox': region.bbox,
                'eccentricity': region.eccentricity,
                'extent': region.extent,
                'orientation': region.orientation,
                'major_axis_length': region.major_axis_length,
                'minor_axis_length': region.minor_axis_length,
                'mean_intensity': region.mean_intensity
            }
            features_list.append(features)
        
        return features_list

        