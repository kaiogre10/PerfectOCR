# PerfectOCR/core/workflow/preprocessing/feature_analyzer.py
from sklearnex import patch_sklearn
patch_sklearn()
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import shannon_entropy
from skimage.filters import rank
from skimage.morphology import disk
import numpy as np

class ImageFeatureAnalyzer:
    
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