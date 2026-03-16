# core/workers/ocr/semantic_clasificator.py
import logging
import numpy as np
from typing import Dict, Any, List, Tuple
from core.utils.math_utils import text_encode
from core.utils.text_utils import find_umd, contains_quantitative
# from core.utils.data_utils import CHAR_NUM

logger = logging.getLogger(__name__)
            
def clasify_words(polygons: Dict[str, Any], worker_config: Dict[str, Any] ) -> Dict[str, Tuple[int | List[int], float]]:
    semantic_range: Tuple[float, float] = worker_config["semantic_range"]
    encode_mean: Tuple[float, float] = worker_config["encode_mean"]
    morph_mean: Tuple[float, float] = worker_config["morph_mean"]
    # char_num = CHAR_NUM

    final_results: Dict[str, Tuple[int | List[int], float]] = {}

    def classify_token(s: str) -> Tuple[int, int]:
        total = len(s)
        total_cuant = int(sum(1 for ch in s if ch.isdigit() or ch in {"$", ","}) / total) if total > 0 else 0
        pct = total_cuant * 100.0 if total_cuant > 0 else 0.0
        
        encoded_text = text_encode(s, ["all"])
        means = np.mean(encoded_text, axis=1).astype(np.float32)
        
        poly_mean = means[0]
        inv_poly_mean = means[1]
        poly_morph_mean = means[2]

        if contains_quantitative(s):
            return 2, total_cuant

        elif find_umd(s):
            return -2, total_cuant
        
        elif morph_mean[1] < poly_morph_mean and poly_mean < encode_mean[0] and encode_mean[1] < inv_poly_mean and semantic_range[1] < pct:
            return 1, total_cuant  # numeric
        elif semantic_range[0] < pct < semantic_range[1] and morph_mean[0] < poly_morph_mean < morph_mean[1]:
            return -1, total_cuant  # code
        
        return 0, total_cuant  # descriptive

    for pid, polygon in polygons.items():
        s = polygon.ocr_text or ""
        if not s:
            continue

        # Fast Path 1: Alfabético puro (mismo resultado para todos los tokens)
        if s.replace(' ', '').isalpha():
            tokens = s.split(' ')
            result = [0] * len(tokens) if len(tokens) > 1 else 0
            # Alfabético puro tiene 0.0 total_cuant en cada token (100% de 0.0 es 0.0)
            final_results[pid] = (result, 0)
            continue

        # Fast Path 2: Numérico puro (mismo resultado para todos los tokens)
        elif s.replace(' ', '').isdigit():
            tokens = s.split(' ')
            result = [1] * len(tokens) if len(tokens) > 1 else 1
            c = len(s)
            # Numérico puro tiene 1.0 total_cuant en cada token. Si hay múltiples tokens, la suma es len(tokens)
            final_results[pid] = (result, c)
            continue

        # Fast Path 3: UMD (Solo si es palabra única)
        elif ' ' not in s and find_umd(s):
            c = (sum(1 for ch in s if ch.isdigit() or ch in {"$", ","}))
            final_results[pid] = (-2, c) # Ajusta 0.0 según corresponda a la palabra
            continue

        elif ' ' not in s and contains_quantitative(s):
            # Requiere un pequeño cálculo manual de cuántitativos aquí:
            c = (sum(1 for ch in s if ch.isdigit() or ch in {"$", ","}))
            final_results[pid] = (2, c)
            continue

        # Procesamiento normal si no entró en los Fast Paths
        elif ' ' in s:
            tokens = s.split(' ')
            token_classes: List[int] = []
            poly_total_cuant = 0
            for t in tokens:
                t_cuant, t_class = classify_token(t)
                token_classes.append(t_cuant)
                poly_total_cuant += t_cuant
            final_results[pid] = (token_classes, poly_total_cuant)
        else:
            t_cuant, t_class = classify_token(s)
            final_results[pid] = (t_cuant, t_class)
    return final_results