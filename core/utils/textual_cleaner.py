# PerfectOCR/core/workers/ocr/text_cleaner.py
import logging
import re
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)

chars = [
        ")", "(", "]", "[", "{", "}", "|", "*", "^", "@",
        "-", "~", "_", "+", "=", "<", ">", ";", ":",
        "'", "!", "¡", "?", "¿", "'", "/", "\\", "''",
    ]

    # normalizar a conjunto de caracteres de longitud 1
drop_single_chars = set(c for c in chars if c and len(c) == 1)
                        
def cleanning_text(worker_config: Dict[str, Any], text_ocr: str, sc: Tuple[str, bool], frecuency_char: Dict[str, float], ocr_confidence: float) -> Optional[str]:
    min_confidence: float  = worker_config.get("min_confidence", {})
    numeric = getattr(sc[0] , "numeric", False) or None
    quantitative = getattr(sc[0] , "quantitative", False) or None
    umd = getattr(sc[0] , "umd", False) or None
            
    low_text: str = filter_low_prob_tokens(worker_config, text_ocr, ocr_confidence, sc, frecuency_char)

    text = remove_special_chars(low_text)

    if (not text.strip() or (ocr_confidence < min_confidence and not (numeric or quantitative or umd)) or re.fullmatch(r'[\s\.\-_,;:]+', text)):
        logger.debug(f"Texto: {text}, conf: {ocr_confidence}")

    # 3. Ruta Normal: Limpiar texto del polígono vacío
    cleaned_text = process_single_text(text)
    if cleaned_text:
        
        return cleaned_text
    else:
        return text
    
def process_single_text(text: str) -> str:
    """
    Limpia una única cadena de texto, aplicando un tratamiento diferenciado
    y seguro a los valores que parecen numéricos.
    """ 
    
    # Dividir por espacios para procesar token por token, preservando la estructura.
    words = text.split(' ')
    processed_words: List[str] = []

    for token in words:
        if not token.strip():  # Evitar procesar tokens vacíos
            processed_words.append(token)
            continue
        
        # Eliminar tokens que sean un carácter especial especificado (ej. ")")
        if is_stray_single_special(token):
            # logger.debug(f"Eliminado unico: '{token}' in {polygon.polygon_id if polygon else ''}")
            continue
    
        else:
            processed_words.append(token)
    
    return ' '.join(processed_words)

def filter_low_prob_tokens(worker_config: Dict[str, Any], text: str, ocr_confidence: float, sc: Tuple[str, bool], freq_norm: Dict[str, float]) -> str:
    min_char = int(worker_config.get("min_char", {}))
    min_probability = float(worker_config.get("min_probability", {}))
    min_confidence = worker_config.get("min_confidence", {})
    numeric = getattr(sc[0] , "numeric", False) or None
    quantitative = getattr(sc[0] , "quantitative", False) or None
    umd = getattr(sc[0] , "umd", False) or None
    try:
        if ocr_confidence and ocr_confidence > min_confidence:
            return text
            
        if numeric or quantitative or umd:
            return text

        tokens = text.split(' ')
        kept: List[str] = []
        removed = 0
        total = 0
        for tok in tokens:
            t = tok.strip()
            if not t:
                kept.append(tok)
                continue

            total += 1

            if any(ch.isdigit() for ch in t):
                kept.append(tok)
                continue

            eff_len = len(''.join(ch for ch in t if not ch.isspace()))
            if eff_len <= min_char:
                score = token_freq_score(freq_norm, t)
                
                if score < min_probability:
                    removed += 1
                    logger.debug(f"Texto:'{t}' | Probabilidad: {score:.4f}")
                    continue
                kept.append(tok)
            else:
                kept.append(tok)

        out = ' '.join(kept)
        if removed > 0:
            logger.info(f"Texto: '{text}' => '{out}'")
        return out
    
    except Exception as e:
        logger.error(f"Error eliminando tokens por frecuencia: {e}", exc_info=True)
        return text

def normalize_char_for_freq(ch: str, frecuency_char: Dict[str, float]) -> str:
        # Mantén tildes/ñ si existen en la tabla; si no, haz fallback a su base
    if ch in get_frecuency_norm(frecuency_char):
        return ch
    base_map = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
        "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U",
        "ü": "u", "Ü": "U", "ñ": "n", "Ñ": "N",
    }
    return base_map.get(ch, ch)

def is_stray_single_special(token: str) -> bool:
    """
    True si el token (tras strip) es exactamente un carácter y está en la lista
    configurada de caracteres a eliminar cuando aparecen aislados.
    """ 
    t = token.strip()
    return len(t) == 1 and t in drop_single_chars

def get_frecuency_norm(frecuency_char: Dict[str, float]) -> Dict[str, float]:

    try:
        
        max_val = max(frecuency_char.values())

        freq_norm: Dict[str, float] = {char: (val / max_val) * 100 for char, val in frecuency_char.items()}
        return freq_norm
    
    except Exception as e:
        logger.error(f"Error al obtener frecuencias normalizadas: {e}", exc_info=True)
        return {}

def token_freq_score(freq_norm: Dict[str, float], token: str) -> float:
    if not freq_norm:
        return 100.0  # si no hay tabla, no castigues
    
    letters: List[float] = []
    for ch in token:
        if ch.isalpha():
            norm = normalize_char_for_freq(ch.lower(), freq_norm)
            if norm in freq_norm:
                letters.append(freq_norm[norm])
            elif norm.isalpha():
                letters.append(0.0)
    if not letters:
    
        return 100.0  # tokens sin letras no se filtran por frecuencia
    return sum(letters) / float(len(letters))

def remove_special_chars(text: str) -> str:
    """
    Elimina todos los caracteres especiales, tanto solitarios como en secuencia.
    Preserva dígitos, letras y espacios.
    """
    if not text:
        return text

    special_chars = drop_single_chars
    if not special_chars:
        logger.warning("Usando patron regex")
        pattern = r'[^A-Za-z0-9\s$¢.,\/\\]'
    else:
        # escapamos los caracteres especiales para regex
        chars_escaped = re.escape("".join(special_chars))
        pattern = r'[' + chars_escaped + r']'

    cleaned = re.sub(pattern, '', text)
    return cleaned
