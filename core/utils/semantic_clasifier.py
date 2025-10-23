# core/workers/ocr/semantic_clasificator.py
import logging
import numpy as np
from typing import Dict, Any, Tuple
from core.utils.text_encoder import encode_text, get_morphological_map
from core.utils.pattern_finder import find_umd, find_quantitative, contains_quantitative

logger = logging.getLogger(__name__)

def clasify_words(s: str, encoder: Dict[str, float], worker_config: Dict[str, Any]) -> Tuple[str, bool]:
    char_num = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", ",", "$"]
    
    semantic_range: Tuple[float, float] = worker_config.get("semantic_range", [])
    encode_mean: Tuple[float, float] = worker_config.get("encode_mean", [])
    morph_mean: Tuple[float, float] = worker_config.get("morph_mean", [])
    
    chars = [ch for ch in s if not ch.isspace()]
    total = len(chars)
    pct = (sum(1 for ch in chars if ch in char_num) / total) * 100.0 if total else 0.0

    encoded_poly = encode_text(s, encoder)
    morph_text = get_morphological_map(s)
    poly_mean = np.mean(encoded_poly)
    poly_morph_mean = np.mean(morph_text) if morph_text else - 1.0
    poly_std = np.std(encoded_poly)
    poly_var = np.var(encoded_poly)

    # Crear objeto SemanticClassification con todos los campos en False
    # semantic_clasification: Dict[str, bool] = {
    #     "numeric": False,
    #     "descriptive": False,
    #     "code": False,
    #     "umd": False,
    #     "quantitative": False,
    # }

    if contains_quantitative(s):
        logger.debug(f"'{s}'| mean: {poly_mean:.4f}, std: {poly_std:.4f}, var: {poly_var:.4f}, morph: {poly_morph_mean}, {pct}% | QUANTITATIVE (pattern_finder)")
        # semantic_clasification = {"quantitative": True}
        return "quantitative", True
        
    # UMD antes (patrones fuertes/ortogonales)
    elif find_umd(s):
        logger.debug(f"'{s}'| mean: {poly_mean:.4f}, std: {poly_std:.4f}, var: {poly_var:.4f}, morph: {poly_morph_mean}, {pct}% | UMD")
        # semantic_clasification = {"umd": True}
        return "umd", True
        
    # Numérico primero; cuantitativo solo si es numérico
    elif semantic_range[1] < pct and poly_mean < encode_mean[0] and morph_mean[1] < poly_morph_mean:
        
        # Verificación cuantitativa SOLO dentro de los numéricos
        has_quantitative = find_quantitative(s)
        if has_quantitative:
            logger.debug(f"'{s}'| mean: {poly_mean:.4f}, std: {poly_std:.4f}, var: {poly_var:.4f}, morph: {poly_morph_mean}, {pct}% | QUANTITATIVE")
            # semantic_clasification = {"quantitative": True}
            return "quantitative", True
            
        else:
            logger.debug(f"'{s}'| mean: {poly_mean:.4f}, std: {poly_std:.4f}, var: {poly_var:.4f}, morph: {poly_morph_mean}, {pct}% | NUMERIC")
            # semantic_clasification = {"numeric": True}
            return "numeric", True
            
    elif pct < semantic_range[0] and poly_morph_mean < morph_mean[0]:
        logger.debug(f"'{s}'| mean: {poly_mean:.4f}, std: {poly_std:.4f}, var: {poly_var:.4f}, morph: {poly_morph_mean}, {pct}% | DESCRIPTIVE")
        # semantic_clasification = {"descriptive": True}
        return "descriptive", True
        
    else:
        logger.debug(f"'{s}' | mean: {poly_mean:.4f}, std: {poly_std:.4f}, var: {poly_var:.4f}, morph: {poly_morph_mean}, {pct}% | CODE")
        # semantic_clasification = {"code": True}
        return "code", True
