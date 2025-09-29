# core/utils/semantic_classifier.py
from typing import Dict, Any

def classify_token(token: str, config: Dict[str, Any]) -> str:
    """
    Clasifica un token usando la misma lógica que semantic_clasificator.
    """
    if not token.strip():
        return "descriptive"
    
    chars = [ch for ch in token if not ch.isspace()]
    total = len(chars)
    
    if total == 0:
        return "descriptive"
    
    digits = sum(1 for ch in chars if ch.isdigit())
    pct = (digits / total) * 100.0
    
    # Usar los mismos rangos configurados
    semantic_config = config.get("semantic_clasificator", {})
    numeric_range = semantic_config.get("numeric", [70.0, 100.0])
    code_range = semantic_config.get("code", [31.0, 69.9])
    
    def norm(r):
        return (min(r[0], r[1]), max(r[0], r[1]))
    
    n_min, n_max = norm(numeric_range)
    c_min, c_max = norm(code_range)
    
    if n_min <= pct <= n_max:
        return "numeric"
    elif c_min <= pct <= c_max:
        return "code"
    else:
        return "descriptive"

def is_numeric(token: str, config: Dict[str, Any]) -> bool:
    """Determina si un token es numérico."""
    return classify_token(token, config) == "numeric"

def is_code(token: str, config: Dict[str, Any]) -> bool:
    """Determina si un token es código."""
    return classify_token(token, config) == "code"

def is_descriptive(token: str, config: Dict[str, Any]) -> bool:
    """Determina si un token es descriptivo."""
    return classify_token(token, config) == "descriptive"