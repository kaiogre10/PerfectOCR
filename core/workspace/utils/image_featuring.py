# PerfectOCR/core/workspace/utils/image_featuring.py
from sklearnex import patch_sklearn
patch_sklearn
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import shannon_entropy
from skimage.filters import rank
from skimage.morphology import disk

class ImageFeaturer:
    
    @staticmethod
    def get_contrast(image: np.ndarray) -> float:
        glcm = graycomatrix(image, [1], [0], levels=256, symmetric=True, normed=True)
        return graycoprops(glcm, 'contrast')[0, 0]
    
    @staticmethod
    def get_dissimilarity(image: np.ndarray) -> float:
        glcm = graycomatrix(image, [1], [0], levels=256, symmetric=True, normed=True)
        return graycoprops(glcm, 'dissimilarity')[0, 0]
    
    @staticmethod
    def get_homogeneity(image: np.ndarray) -> float:
        glcm = graycomatrix(image, [1], [0], levels=256, symmetric=True, normed=True)
        return graycoprops(glcm, 'homogeneity')[0, 0]
    
    @staticmethod
    def get_energy(image: np.ndarray) -> float:
        glcm = graycomatrix(image, [1], [0], levels=256, symmetric=True, normed=True)
        return graycoprops(glcm, 'energy')[0, 0]
    
    @staticmethod
    def get_correlation(image: np.ndarray) -> float:
        glcm = graycomatrix(image, [1], [0], levels=256, symmetric=True, normed=True)
        return graycoprops(glcm, 'correlation')[0, 0]
    
    @staticmethod
    def get_entropy(image: np.ndarray) -> float:
        return shannon_entropy(image)
    
    @staticmethod
    def get_mean(image: np.ndarray) -> float:
        return np.mean(image)
    
    @staticmethod
    def get_std(image: np.ndarray) -> float:
        return np.std(image)
    
    @staticmethod
    def get_skewness(image: np.ndarray) -> float:
        return float(np.mean(((image - np.mean(image)) / np.std(image)) ** 3))
    
    @staticmethod
    def get_kurtosis(image: np.ndarray) -> float:
        return float(np.mean(((image - np.mean(image)) / np.std(image)) ** 4))
    
    @staticmethod
    def get_local_entropy(image: np.ndarray) -> float:
        return np.mean(rank.entropy(image, disk(5)))
    
    @staticmethod
    def get_lbp_features(image: np.ndarray) -> list:
        lbp = local_binary_pattern(image, 8, 1, method='uniform')
        lbp_hist = np.histogram(lbp, bins=10)[0]
        return lbp_hist.tolist()
    
# uSO:
#    analyzer = ImageFeatureAnalyzer(gray_image)

# Solo cuando necesites bilateral filter
#if analyzer.get_dissimilarity() > 0.6:
  #  apply_bilateral_filter()

# Solo cuando necesites contraste
#if analyzer.get_std() < 30:
 #   apply_clahe()

# === COMENTARIOS PARA CONFIGURAR FILTROS SEGÚN FEATURES ===

# --- GLCM Features ---
# contrast (diferencia local): Controla intensidad de CLAHE - alto contraste = clip_limit bajo
# dissimilarity (variación): Decide bilateral filter - alta variación = más denoising
# homogeneity (uniformidad): Ajusta grid size CLAHE - alta homogeneidad = grids más grandes
# energy (uniformidad global): Controla sharpening - baja energía = más nitidez
# correlation (dependencia píxeles): Ajusta kernel median filter - baja correlación = kernel más grande

# --- Estadísticos básicos ---
# entropy (información): Controla todos los filtros - alta entropía = procesamiento más agresivo
# mean (brillo): Detecta over/under exposure - ajusta normalización
# std (contraste): Decide si aplicar CLAHE - baja std = necesita contraste
# skewness (asimetría): Detecta documentos mal escaneados - alta asimetría = más corrección
# kurtosis (picos): Detecta ruido sal y pimienta - alta kurtosis = más median filter

# --- Específicos ---
# local_entropy (textura local): Ajusta parámetros de bilateral filter
# lbp_features (patrones): Detecta tipo de documento para configurar pipeline completo