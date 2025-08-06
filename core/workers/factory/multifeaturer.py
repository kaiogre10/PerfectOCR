# PerfectOCR/core/workspace/utils/multifeaturer.py
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import shannon_entropy, label, regionprops
from skimage.filters import threshold_sauvola, unsharp_mask, rank
from skimage.morphology import disk
from joblib import Parallel, delayed
from typing import Dict, Any
import numpy as np
import warnings
import os
import json

# Todas las features en formato lambda uniforme para facilitar mantenimiento

FEATURE_FUNCTIONS = {
    'region':{
        # --- Propiedades básicas de área y forma ---
        'area': lambda region: region.area,
        'bbox': lambda region: region.bbox,
        'bbox_area': lambda region: region.bbox_area,
        'convex_area': lambda region: region.convex_area,
        'filled_area': lambda region: region.filled_area,

        # --- Centroides y coordenadas ---
        'centroid': lambda region: region.centroid,
        'local_centroid': lambda region: region.local_centroid,
        'weighted_centroid': lambda region: region.weighted_centroid,
        'weighted_local_centroid': lambda region: region.weighted_local_centroid,
        'coords': lambda region: region.coords,

        # --- Propiedades geométricas ---
        'eccentricity': lambda region: region.eccentricity,
        'extent': lambda region: region.extent,
        'major_axis_length': lambda region: region.major_axis_length,
        'minor_axis_length': lambda region: region.minor_axis_length,
        'orientation': lambda region: region.orientation,
        'equivalent_diameter': lambda region: region.equivalent_diameter,
        'equivalent_diameter_area': lambda region: region.equivalent_diameter_area,
        'feret_diameter_max': lambda region: region.feret_diameter_max,
        'perimeter': lambda region: region.perimeter,
        'perimeter_crofton': lambda region: region.perimeter_crofton,
        'solidity': lambda region: region.solidity,

        # --- Topología ---
        'euler_number': lambda region: region.euler_number,

        # --- Propiedades de intensidad ---
        'max_intensity': lambda region: region.max_intensity,
        'mean_intensity': lambda region: region.mean_intensity,
        'min_intensity': lambda region: region.min_intensity,

        # --- Momentos estadísticos ---
        'moments': lambda region: region.moments,
        'moments_central': lambda region: region.moments_central,
        'moments_normalized': lambda region: region.moments_normalized,
        'moments_hu': lambda region: region.moments_hu,
        'weighted_moments': lambda region: region.weighted_moments,
        'weighted_moments_central': lambda region: region.weighted_moments_central,
        'weighted_moments_normalized': lambda region: region.weighted_moments_normalized,
        'weighted_moments_hu': lambda region: region.weighted_moments_hu,

        # --- Tensor de inercia ---
        'inertia_tensor': lambda region: region.inertia_tensor,
        'inertia_tensor_eigvals': lambda region: region.inertia_tensor_eigvals,

        # --- Imágenes y metadatos ---
        'image': lambda region: region.image,
        'convex_image': lambda region: region.convex_image,
        'filled_image': lambda region: region.filled_image,
        'intensity_image': lambda region: region.intensity_image,
        'label': lambda region: region.label,
        'slice': lambda region: region.slice
    },
    'image': {    # --- Features de imagen global ---
        'contrast': lambda image: graycoprops(graycomatrix(image, [1], [0], levels=256, symmetric=True, normed=True), 'contrast')[0, 0],
        'shannon_entropy': lambda image: shannon_entropy(image),
        'lbp_hist': lambda image: (lambda hist: hist.astype("float") / (hist.sum() + 1e-6))(np.histogram(local_binary_pattern(image, P=8, R=1, method='uniform').ravel(), bins=np.arange(0, 11), range=(0, 10))[0]),
            # --- Filtros de scikit-image ---
        'threshold_sauvola': lambda image: threshold_sauvola(image),
        'unsharp_mask': lambda image: unsharp_mask(image),
        'rank_mean_disk3': lambda image: rank.mean(image, disk(3)),
        'rank_median_disk3': lambda image: rank.median(image, disk(3)),
        'rank_entropy_disk3': lambda image: rank.entropy(image, disk(3)),
        'rank_enhance_disk3': lambda image: rank.enhance_contrast(image, disk(3)),
    },
    'binary': {
        # --- Features agregados para imágenes binarizadas ---
        'region_count': lambda binary: len(regionprops(label(binary))) if binary.max() > 0 else 0,
        'region_area_total': lambda binary: sum([r.area for r in regionprops(label(binary))]) if binary.max() > 0 else 0,
        'region_area_mean': lambda binary: np.mean([r.area for r in regionprops(label(binary))]) if binary.max() > 0 and regionprops(label(binary)) else 0.0,
        'region_area_max': lambda binary: np.max([r.area for r in regionprops(label(binary))]) if binary.max() > 0 and regionprops(label(binary)) else 0.0,
        'region_area_min': lambda binary: np.min([r.area for r in regionprops(label(binary))]) if binary.max() > 0 and regionprops(label(binary)) else 0.0,
        'region_perimeter_mean': lambda binary: np.mean([r.perimeter for r in regionprops(label(binary))]) if binary.max() > 0 and regionprops(label(binary)) else 0.0,
        'region_perimeter_max': lambda binary: np.max([r.perimeter for r in regionprops(label(binary))]) if binary.max() > 0 and regionprops(label(binary)) else 0.0,
        'region_perimeter_min': lambda binary: np.min([r.perimeter for r in regionprops(label(binary))]) if binary.max() > 0 and regionprops(label(binary)) else 0.0,
        'region_euler_mean': lambda binary: np.mean([r.euler_number for r in regionprops(label(binary))]) if binary.max() > 0 and regionprops(label(binary)) else 0.0,
        'region_extent_mean': lambda binary: np.mean([r.extent for r in regionprops(label(binary))]) if binary.max() > 0 and regionprops(label(binary)) else 0.0,
        'region_solidity_mean': lambda binary: np.mean([r.solidity for r in regionprops(label(binary))]) if binary.max() > 0 and regionprops(label(binary)) else 0.0,
        'region_eccentricity_mean': lambda binary: np.mean([r.eccentricity for r in regionprops(label(binary))]) if binary.max() > 0 and regionprops(label(binary)) else 0.0,
        'region_density': lambda binary: len(regionprops(label(binary))) / (binary.shape[0] * binary.shape[1]) if binary.max() > 0 else 0.0,
        'white_pixel_ratio': lambda binary: np.sum(binary > 0) / (binary.shape[0] * binary.shape[1]),
        'region_labels': lambda binary: [r.label for r in regionprops(label(binary))] if binary.max() > 0 else []
    },
}

# FUNCIÓN INTERNA: RECOLECTA UNA FEATURE PARA TODOS LOS OBJETOS

def _collect_feature_for_all(objs, func: Dict[str,]):
    """
    Calcula una feature específica para todos los objetos en paralelo.
    """
    return Parallel(n_jobs=n_jobs)(
        delayed(func)(obj) for obj in objs
    )

# FUNCIÓN PÚBLICA: PUNTO DE ACCESO ÚNICO PARA PEDIDOS DE FEATURES

class FeatureFactory:
    def __init__(self, project_root: str):
        self.project_root = project_root
        self._density_encoder = None  # ← Lazy loading del JSON
        
    def request_features(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estructura simple con carga automática de JSON cuando se necesita.
        """
        objects = request_data.get("objects", [])
        features = request_data.get("features", [])
        
        if not objects or not features:
            return {}
        
        results = {}
        for feature_name in features:
            func = self._find_specific_function(feature_name)
            if func:
                results[feature_name] = self._calculate_single_feature(objects, func)
        
        return results
    
    def _find_specific_function(self, feature_name: str):
        """Busca función específica con carga automática de JSON."""
        
        # Features de imagen (sin JSON)
        image_features = {
            'area': lambda region: region.area,
            'perimeter': lambda region: region.perimeter,
            'eccentricity': lambda region: region.eccentricity,
            'shannon_entropy': lambda image: shannon_entropy(image),
            # ... más features de imagen
        }
        
        # Features de vectorización (requieren JSON)
        vectorization_features = {
            'density_encode': self._density_encode_feature,
            'character_frequency': self._character_frequency_feature,
            'text_complexity': self._text_complexity_feature,
            # ... más features de vectorización
        }
        
        # Buscar en features de imagen
        if feature_name in image_features:
            return image_features[feature_name]
        
        # Buscar en features de vectorización
        if feature_name in vectorization_features:
            return vectorization_features[feature_name]
        
        return None
    
    def _load_density_encoder(self):
        """Carga JSON solo cuando se necesita."""
        if self._density_encoder is None:
            json_path = os.path.join(self.project_root, "core/workers/factory/density_enconder.json")
            with open(json_path, 'r', encoding='utf-8') as f:
                self._density_encoder = json.load(f)
            logger.info("Density encoder JSON cargado en memoria")
    
    def _density_encode_feature(self, text_obj):
        """Feature que requiere JSON cargado."""
        self._load_density_encoder()  # ← Carga automática
        # Lógica de densidad usando self._density_encoder
        return self._calculate_density(text_obj.text, self._density_encoder)
    
    def _character_frequency_feature(self, text_obj):
        """Feature que requiere JSON cargado."""
        self._load_density_encoder()  # ← Carga automática
        # Lógica de frecuencia usando self._density_encoder
        return self._calculate_frequency(text_obj.text, self._density_encoder)