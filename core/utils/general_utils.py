# core/workers/ocr/semantic_clasificator.py
import logging
import numpy as np
from typing import Dict, Any, List, Tuple
from core.utils.math_utils import text_encode
from core.utils.text_utils import find_umd, contains_quantitative
from core.utils.data_utils import CHAR_NUM

logger = logging.getLogger(__name__)
            
def clasify_words(polygons: Dict[str, Any], worker_config: Dict[str, Any] ) -> Dict[str, int | List[int]]:
    # t0 = time.perf_counter()
    semantic_range: Tuple[float, float] = worker_config["semantic_range"]
    encode_mean: Tuple[float, float] = worker_config["encode_mean"]
    morph_mean: Tuple[float, float] = worker_config["morph_mean"]

    final_results: Dict[str, int | list[int]] = {}

    def classify_token(s: str) -> int:
        total = len(s)
        pct = (sum(1 for ch in s if ch in CHAR_NUM) / total) * 100.0 if total else 0.0
        
        encoded_text = text_encode(s, ["all"])
        means = np.mean(encoded_text, axis=1).astype(np.float32)
        
        poly_mean = means[0]
        inv_poly_mean = means[1]
        poly_morph_mean = means[2]

        if contains_quantitative(s):
            return 2

        elif find_umd(s):
            return -2
        
        elif morph_mean[1] < poly_morph_mean and poly_mean < encode_mean[0] and encode_mean[1] < inv_poly_mean and semantic_range[1] < pct:
            return 1  # numeric
        elif semantic_range[0] < pct < semantic_range[1] and morph_mean[0] < poly_morph_mean < morph_mean[1]:
            return -1  # code
        
        return 0  # descriptive

    for pid, polygon in polygons.items():
        s = polygon.ocr_text or ""
        if not s:
            continue

        # Fast Path 1: Alfabético puro (mismo resultado para todos los tokens)
        if s.replace(' ', '').isalpha():
            tokens = s.split(' ')
            final_results[pid] = [0] * len(tokens) if len(tokens) > 1 else 0
            continue

        # Fast Path 2: Numérico puro (mismo resultado para todos los tokens)
        elif s.replace(' ', '').isdigit():
            tokens = s.split(' ')
            final_results[pid] = [1] * len(tokens) if len(tokens) > 1 else 1
            continue

        # Fast Path 3: UMD (Solo si es palabra única)
        elif ' ' not in s and find_umd(s):
            final_results[pid] = -2
            continue

        elif ' ' not in s and contains_quantitative(s):
            final_results[pid] = 2
            continue

        # Procesamiento normal si no entró en los Fast Paths
        elif ' ' in s:
            tokens = s.split(' ')
            final_results[pid] = [classify_token(t) for t in tokens]
        else:
            final_results[pid] = classify_token(s)
            
    # logger.info(f"Clasificación semantica completa en: {time.perf_counter() - t0:.6f}'s")
    return final_results
