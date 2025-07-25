# PerfectOCR/core/workspace/utils/multifeaturer.py
from sklearnex import patch_sklearn
patch_sklearn()

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import shannon_entropy, label, regionprops
from skimage.filters import rank
from skimage.morphology import disk
from joblib import Parallel, delayed
import numpy as np
import warnings
import mkl

mkl.set_num_threads(4)

# ------------------------------------------------------------------------------
# DICCIONARIO GLOBAL DE FUNCIONES DE FEATURES
# Todas las features en formato lambda uniforme para facilitar mantenimiento
# ------------------------------------------------------------------------------
FEATURE_FUNCTIONS = {
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
    'slice': lambda region: region.slice,

    # --- Features de imagen global ---
    'contrast': lambda image: graycoprops(graycomatrix(image, [1], [0], levels=256, symmetric=True, normed=True), 'contrast')[0, 0],
    'shannon_entropy': lambda image: shannon_entropy(image),
    'lbp_hist': lambda image: (lambda hist: hist.astype("float") / (hist.sum() + 1e-6))(np.histogram(local_binary_pattern(image, P=8, R=1, method='uniform').ravel(), bins=np.arange(0, 11), range=(0, 10))[0]),

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
    'region_labels': lambda binary: [r.label for r in regionprops(label(binary))] if binary.max() > 0 else [],
}

# ------------------------------------------------------------------------------
# FUNCIÓN INTERNA: RECOLECTA UNA FEATURE PARA TODOS LOS OBJETOS
# ------------------------------------------------------------------------------
def _collect_feature_for_all(objs, func, n_jobs):
    """
    Calcula una feature específica para todos los objetos en paralelo.
    Esta es la función que hace el trabajo pesado de cálculo.
    
    Args:
        objs (list): Lista de objetos (imágenes o regiones).
        func (callable): Función lambda que calcula la feature.
        n_jobs (int): Número de núcleos a usar.
    
    Returns:
        list: Lista con los resultados de la feature para cada objeto.
    """
    return Parallel(n_jobs=n_jobs)(
        delayed(func)(obj) for obj in objs
    )

# ------------------------------------------------------------------------------
# FUNCIÓN PÚBLICA: RECIBE EL PEDIDO Y COORDINA LA RECOLECCIÓN
# ------------------------------------------------------------------------------
def request_features(objs, features, n_jobs=-1):
    """
    Función pública que recibe el pedido de features, valida, y coordina
    el cálculo usando el patrón de rondas.
    
    Args:
        objs (list): Lista de imágenes (np.ndarray) o regiones (regionprops).
        features (list): Lista de nombres de las features a calcular.
        n_jobs (int): Número de núcleos a usar (-1 = todos los disponibles).
    
    Returns:
        dict: Diccionario donde cada clave es una feature y el valor es una
              lista con los resultados para cada objeto.
    
    Raises:
        ValueError: Si alguna feature solicitada no está implementada.
        TypeError: Si los objetos o features no son del tipo correcto.
    """
    # Validaciones de entrada
    if not isinstance(objs, list) or not objs:
        raise TypeError("objs debe ser una lista no vacía.")
    if not isinstance(features, list) or not features:
        raise TypeError("features debe ser una lista no vacía.")
    
    results = {}
    for feat in features:
        func = FEATURE_FUNCTIONS.get(feat)
        if func is None:
            raise ValueError(f"Feature '{feat}' no está implementada. "
                           f"Features disponibles: {list(FEATURE_FUNCTIONS.keys())}")
        
        # Delega el cálculo a la función recolectora
        try:
            results[feat] = _collect_feature_for_all(objs, func, n_jobs)
        except Exception as e:
            # Manejo de errores en el cálculo
            raise RuntimeError(f"Error calculando feature '{feat}': {str(e)}")
    
    return results

# ------------------------------------------------------------------------------
# FUNCIÓN DE CONVENIENCIA: PARA UNA SOLA FEATURE
# ------------------------------------------------------------------------------
def request_single_feature(objs, feature, n_jobs=-1):
    """
    Función de conveniencia para calcular una sola feature.
    
    Args:
        objs (list): Lista de objetos.
        feature (str): Nombre de la feature a calcular.
        n_jobs (int): Número de núcleos a usar.
    
    Returns:
        list: Lista con los resultados para cada objeto.
    """
    result_dict = request_features(objs, [feature], n_jobs)
    return result_dict[feature]