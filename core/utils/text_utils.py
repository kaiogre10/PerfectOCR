import re
import logging
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict, Pattern, Set, Any
from core.utils.math_utils import text_encode
from core.utils.data_utils import SPECIAL_CHARS, NOT_VALID_CHARS, NOT_VALID_PUNT_CHARS, CHAR_NUM, ALONE_CHARS

logger = logging.getLogger(__name__)

# Patrón para secuencias especiales de 2 o más caracteres no alfanuméricos (excluyendo espacio, $, ,)
secuence_pattern: Pattern[str] = re.compile(r'[^a-zA-Z0-9\s$]{2,}')
_sequence_middle_pattern: Pattern[str] = re.compile(r'(?<=[a-zA-Z0-9$])[^a-zA-Z0-9\s$]{2,}(?=[a-zA-Z0-9$])')

_numeric_separator: Pattern[str] =  re.compile(r'^([$\u00A2]?\s*)(-?\d[\d.,]*)(\s*[$\u00A2]?)$', re.IGNORECASE)

_punt_split_pattern: Pattern[str] = re.compile(r'([=;:!?])')

# Espacios múltiples
_spaces_pattern: Pattern[str] = re.compile(r'\s+')

# Terminación
_termination_pattern: Pattern[str] = re.compile(r'(?i)(s|c|r)?i0n\b', re.IGNORECASE)

# Num Teléfonos
# _phone_number: Pattern[str] = re.compile(r'^\d{10}$')

# Siglas/Acrónimos
_acronym_pattern: Pattern[str] = re.compile(r'^(?:(?:[A-Za-z]\.){1,}[A-Za-z]\.?|sa|cv|no)(?:[:;,.])?$', re.IGNORECASE)

# RFC
_rfc_code_pattern: Pattern[str] = re.compile(r'^([A-ZÑ&]{3,4})\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])[A-Z0-9]{3}$', re.IGNORECASE)

# IVA
_iva_pattern: Pattern[str] = re.compile(r'\b(I\.?V\.?A\.?)\b')

_date_long: Pattern[str] = re.compile(r'^\d{8}$')
_date_patterns: List[Pattern[str]] = [
    re.compile(r'\b(ene(ro)?|feb(rero)?|mar(zo)?|abr(il)?|may(o)?|jun(io)?|jul(io)?|ago(sto)?|sep(t(iembre)?)?|oct(ubre)?|nov(iembre)?|dic(iembre)?)\b', re.IGNORECASE),
    re.compile(r'\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b'),
    re.compile(r'\b(0?[1-9]|[12]\d|3[01])[\/\-\.](0?[1-9]|1[0-2])\b'),
    re.compile(r'\b(199\d|20\d{2})\b'),
    re.compile(r'\b([01]?\d|2[0-3]):[0-5]\d(:[0-5]\d)?\b'),
    re.compile(r'\b([01]\d|2[0-3])[0-5]\d\s*[AaPp]\.?[Mm]\.?\b'),
    
]

# UMD
_umd_correct = re.compile(r'(?<=/)[0-9Oo]+(?=\b)', re.IGNORECASE)

_umd_patterns: List[Pattern[str]] = [
    re.compile(r'\b\d+([.,]\d+)?\s*(kg(r)?|kilo(s)?|g(r|ramo(s)?)?|mg|lb(s)?|libra(s)?|oz|onza(s)?|ton(elada(s)?)?)\b', re.IGNORECASE),
    re.compile(r'\b[Cc]\s*/\s*\d+\b'),
    re.compile(r'\b\d+([.,]\d+)?\s*(l(t(r)?)?|litro(s)?|ml|cc|gal(on(es)?)?)\b', re.IGNORECASE),
    re.compile(r'\b\d+([.,]\d+)?\s*(m(t(r)?)?|metro(s)?|cm|mm|km|in|pulg(ada(s)?)?|ft)\b', re.IGNORECASE),
    re.compile(r'\b\d+([.,]\d+)?\s*(m(t)?(2|\^2|²)|cm(2|\^2|²)|km(2|\^2|²))\b', re.IGNORECASE),
    re.compile(r'\b(m(t)?(2|\^2|²)|cm(2|\^2|²)|km(2|\^2|²))\b', re.IGNORECASE),
    re.compile(r'\b\d+\s*/\s*\d+\s*(kg(r)?|kilo(s)?|g(r)?|l(t(r)?)?|litro(s)?|ml|pz(a)?(s)?|ud(s)?)\b', re.IGNORECASE),
    re.compile(r'\b[1-9]\s*/\s*(2|4|8|16|32|64)\b'),
    re.compile(r'\b\d+(?:\s*[xX]\s*\d+){1,}\b', re.IGNORECASE),
]

# Define los patrones como strings
digit_pattern = r"[0-9oO]"
currency_pattern = r"[$¢]"

# Compila los patrones base
digit = re.compile(digit_pattern)
currency = re.compile(currency_pattern)

# Usa los strings en las interpolaciones
_amount_body_pattern = (
    rf"(?:{digit_pattern}+(?:[.,]{digit_pattern}+)?|"
    rf"{digit_pattern}{{1,3}}(?:[.,]{digit_pattern}{{3}})*)(?:[.,]{digit_pattern}+)?"
)

_token_pattern = (
    rf"{currency_pattern}\s*{_amount_body_pattern}|"
    rf"{_amount_body_pattern}\s*{currency_pattern}|"
    rf"{_amount_body_pattern}"
)
_token = re.compile(_token_pattern)

_start_pattern = rf"^{currency_pattern}\s*{_amount_body_pattern}$"
_start = re.compile(_start_pattern, re.IGNORECASE)

_middle_pattern = rf"^{_amount_body_pattern}\s*{currency_pattern}\s*{_amount_body_pattern}$"
_middle = re.compile(_middle_pattern, re.IGNORECASE)

_end_pattern = rf"^{_amount_body_pattern}\s*{currency_pattern}$"
_end = re.compile(_end_pattern, re.IGNORECASE)

_multi_pattern = rf"^(?:\s*{currency_pattern}\s*{_amount_body_pattern}\s*){{2,}}$"
_multi = re.compile(_multi_pattern, re.IGNORECASE)

_decimal = re.compile(r"^(?:\d+|\d{1,3}(?:[.,]\d{3})+)[.,]\d{2,}$", re.IGNORECASE)
_digits = re.compile(r"\d+")
_split_pattern = rf"{currency_pattern}\s*{_amount_body_pattern}"
_split = re.compile(_split_pattern)

def _build_quantitative_patterns() -> Dict[str, Pattern[str]]:
    return {
        "currency":     currency,
        "token":        _token,
        "start":        _start,
        "middle":       _middle,
        "end":          _end,
        "multi":        _multi,
        "decimal":      _decimal,
        "digits":       _digits,
        "split":        _split,
    }
_Q = _build_quantitative_patterns()

char_num = CHAR_NUM
valid_chars = ALONE_CHARS
special_chars = SPECIAL_CHARS
not_valid_chars = NOT_VALID_CHARS
not_valid_punt_chars = NOT_VALID_PUNT_CHARS
char_num_point = char_num.copy()
char_num_point.add(".")
        
def termination_detect(text: str) -> bool:
    if not validate_text(text):
        return False
    return bool(_termination_pattern.search(text))

def is_acronym(text: str) -> bool:
    try:
        if not text:
            return False
        if _acronym_pattern.search(text):
            return True
    except Exception as e:
        logger.error(f"Error buscando siglas: {e}", exc_info=True)
    return False

def find_umd(s: str) -> bool:
    try:
        if not s:
            return False
        if any(p.search(s) for p in _umd_patterns):
            return True        
    except Exception as e:
        logger.warning(f"Error buscando unidades de medida: {e}", exc_info=True)
    return False

def find_date(s: str) -> bool:
    try:
        if not s:
            return False
        if not any(c.isdecimal() for c in s):
            return False

        # Filtro rápido: si todos los caracteres son cuantitativos, no es fecha
        s = s.replace(" ", "")
        
        if set(s).issubset(char_num_point):
            return False

        # Validación especial para fechas tipo DDMMAAAA (8 dígitos)
        # Extraer fecha solo con regex, sin librerías externas
        # date_match = _date_long.search(s)
        # if date_match is not None:
        #     logger.info(f"FECHAS: {s}: {date_match}")
        #     return True
        # else:
        is_date = any(p.search(s) for p in _date_patterns)
        is_umd = any(p.search(s) for p in _umd_patterns)

        # Solo es fecha si coincide con fecha y NO con UMD
        if is_date and not is_umd:
            return True
        return False
    except TypeError as e:
        logger.error(f"Error buscando fecha: {e}", exc_info=True)
    return False
    
def find_rfc(s: str) -> bool:
    try:
        if not s:
            return False
        if not any(c.isdecimal() for c in s):
            return False
        return bool(_rfc_code_pattern.search(s))
    except Exception as e:
        logger.warning(f"Error buscando RFC: {e}", exc_info=True)
        return False

def find_iva(s: str) -> bool:
    try:
        if not s:
            return False
        return bool(_iva_pattern.search(s))
    except Exception as e:
        logger.warning(f"Error buscando IVA: {e}", exc_info=True)
    return False

def contains_quantitative(s: str) -> bool:
    """
    Devuelve True si hay al menos un cuantitativo válido en cualquier parte del texto.
    """
    runs = find_quantitative_runs(s)
    return len(runs) > 0

def is_quantitative(s: str) -> bool:

    s_norm = s.replace("o", "0").replace("O", "0")

    # OCR típico: S1275.04 -> $1275.04
    if len(s_norm) > 1 and s_norm[0] in "Ss" and s_norm[1].isdecimal():
        s_norm = "$" + s_norm[1:]

    currency_symbols = "$¢"
    for sym in currency_symbols:
        idx = s_norm.find(sym)
        if idx != -1:
            after = s_norm[idx+1:]
            if any(c.isdecimal() for c in after):
                maybe_amt = after.lstrip()
                possible_num = "".join(ch for ch in maybe_amt if ch.isdecimal() or ch in ".,")
                if possible_num == "00":
                    return False
                if idx == len(s_norm) - 1:
                    return False
                if idx == 0 or not s_norm[:idx].strip().isdecimal():
                    break

    if _Q["end"].match(s_norm):
        return False

    amounts = _Q["digits"].findall(s_norm)
    if not (s_norm.endswith('.00') or s_norm.endswith(',00')):
        if any(c == "00" for c in amounts if len(amounts) > 1 or c != "00"):
            return False

    return bool(
        _Q["start"].match(s_norm) or
        _Q["middle"].match(s_norm) or
        _Q["multi"].match(s_norm) or
        _Q["decimal"].match(s_norm)
    )

def find_quantitative_runs(s: str) -> List[Tuple[int, int, str]]:
    runs: List[Tuple[int, int, str]] = []
    s_norm = s.replace("o", "0").replace("O", "0")

    for m in _Q["token"].finditer(s_norm):
        tok = m.group(0)
        if is_quantitative(tok):
            runs.append((m.start(), m.end(), s[m.start():m.end()]))

    currency_count = sum(1 for _, _, tok in runs if _Q["currency"].search(tok))
    if currency_count > 1:
        split_runs: List[Tuple[int, int, str]] = []
        for match in _Q["split"].finditer(s_norm):
            split_runs.append((match.start(), match.end(), s[match.start():match.end()]))
        return split_runs
    return runs

def separate_punt(text: str) -> str:
    if is_acronym(text):
        return text
    else:
        parts = _punt_split_pattern.split(text)
        return " ".join(p.strip() for p in parts if p.strip() and not _punt_split_pattern.fullmatch(p))
    
def remove_special_sequences(text: str) -> str:
    cleaned = _sequence_middle_pattern.sub(' ', text)
    cleaned = secuence_pattern.sub('', cleaned)
    return cleaned.strip()

def validate_text(text: str) -> bool:
    # Esta es la forma más rápida en Python puro (C-API)
    # 1. 'if not text' captura None y "" ultrarrápido.
    # 2. '.strip()' es más rápido que 'isspace' si el string tiene contenido
    # porque 'if s.strip()' es una sola operación de verdad en C.
    return bool(text and text.strip())
        
def valid_punt_chars() -> Set[str]:
    return not_valid_punt_chars.union(not_valid_chars)

def validate_alone_chars(text: str) -> bool:
    """Valida si un caracter solitario es válido o es ruido"""
    text = text.strip()
    if len(text) > 1:
       return True

    elif text.isdecimal():
        return True
    
    elif text.lower() in valid_chars:
        return True

    else:
        return False

def is_upper(text: str) -> bool:
    uppers: int = 0
    text = text.strip()
    for char in text:
        if char.islower():
            continue
        uppers += 1
    upper_mean = uppers/len(text)
    # print(f"{upper_mean}")
    if upper_mean > 0.666:
        return True
    else:
        return False

def estandarice_uppers_lowers(text_base: str, clean_text: str) -> str:
    if text_base.isupper() or is_upper(text_base):
        return clean_text.upper()
    elif text_base.islower():
        return clean_text.lower()
    elif text_base.istitle():
        return clean_text.title()
    else:
        return clean_text

def space_removal(text: str) -> str:
    """
    Limpia espacios múltiples y espacios iniciales/finales de un texto.
    Reemplaza múltiples espacios consecutivos por un solo espacio y elimina espacios al inicio y final.
    """
    if not text:
        return ""
    
    # Reemplaza cualquier secuencia de espacios (\s+) por uno solo y limpia bordes
    clean_text = _spaces_pattern.sub(" ", text).strip()    
    return clean_text if validate_text(clean_text) else ""

def detect_special_strings(text: str) -> bool:
    """
    Retorna True si el texto NO contiene ningún carácter alfanumérico.
    Maneja correctamente caracteres acentuados y latinos (Unicode).
    """
    if not text:
        return True
    
    return not any(char.isalnum() for char in text)

def numeric_separator(token: str) -> str:
    """
    Normaliza separadores numéricos en montos cuantitativos.

    Ejemplo:
        3.226.66 -> 3,226.66
    """
    try:
        if not validate_text(token):
            return token

        t = token.strip()
        if not is_quantitative(t):
            return token

        match = _numeric_separator.match(t)
        if not match:
            return token

        prefix, number, suffix = match.groups()
        separators = [i for i, ch in enumerate(number) if ch in ".,"]
        if not separators:
            return token

        last_sep = separators[-1]
        frac = number[last_sep + 1 :]
        if not (frac.isdecimal() and len(frac) == 2):
            return token

        int_digits = number[:last_sep].replace(".", "").replace(",", "")

        sign = ""
        if int_digits.startswith("-"):
            sign = "-"
            int_digits = int_digits[1:]

        if not int_digits:
            return token

        groups: List[str] = []
        while len(int_digits) > 3:
            groups.append(int_digits[-3:])
            int_digits = int_digits[:-3]
        groups.append(int_digits)

        int_grouped = ",".join(reversed(groups))
        return f"{prefix}{sign}{int_grouped}.{frac}{suffix}"

    except Exception as e:
        logger.warning(f"Error normalizando separadores numéricos: {e}", exc_info=True)
        return token
        
def clasify_words(polygons: Dict[str, Any], worker_config: Dict[str, Any] ) -> Dict[str, Tuple[List[int], int]]:
    density_mean: Tuple[float, float] = worker_config["encode_mean"]
    morph_mean: Tuple[float, float] = worker_config["morph_mean"]

    final_results: Dict[str, Tuple[List[int], int]] = {}

    def classify_token(s: str) -> Tuple[int, int]:
        if not s:
            return (0, 0)
        
        elif s.isalpha():
            return (0, 0)
    
        elif s.isdecimal():
            return (1, len(s))

        total_cuant = sum(1 for ch in s if ch in CHAR_NUM)

        if total_cuant == 0:
            if find_umd(s):
                return (-2, total_cuant)
            else:
                return (0, 0)
        
        elif set(s).issubset(char_num_point):
            return (2, len(s))
        
        elif contains_quantitative(s):
            return (2, total_cuant)
            
        elif find_umd(s):
            return (-2, total_cuant)
        
        elif s.isalnum():
            if all(c.isupper() for c in s if c.isalpha()):
                # logger.info(f"Texto code: {s}")
                return (-1, total_cuant)
            else:
                return (0, 0)

        encoded_text = text_encode(s, ["density", "morphological"])
        text_means = np.mean(encoded_text, axis=1, dtype=np.float32)

        if text_means[0] > density_mean[1]:
            return (0, 0)
        
        elif text_means[0] < density_mean[0]:
            if contains_quantitative(s):
                return (2, total_cuant)
            else:
                return (1, total_cuant)  # numeric
        
        elif text_means[0] < density_mean[1] and text_means[1] > morph_mean[0]:
            return (-1, total_cuant)  # code

        else: 
            return (0, total_cuant)  # descriptive

    for pid, polygon in polygons.items():
        s = polygon.ocr_text or ""
        s = s.strip()
        if not s:
            continue

        tokens = s.split()
        total_tokens = len(tokens)
        if not tokens:
            continue
        
        elif 0 >= total_tokens:
            continue

        elif total_tokens > 1:
            if "".join(tokens).isalpha():
                result = [0] * total_tokens
                final_results[pid] = (result, 0)
                continue

            token_classes: List[int] = []
            poly_total_cuant = 0
            for t in tokens:
                t_class, t_cuant = classify_token(t.strip())
                token_classes.append(t_class)
                poly_total_cuant += t_cuant
            final_results[pid] = (token_classes, poly_total_cuant)
        
        else:
            if s.isalpha():
                final_results[pid] = ([0], 0)
                continue

            elif s.isdecimal():
                final_results[pid] = ([1], len(s))
                continue

            elif set(s).issubset(char_num_point):
                final_results[pid] = ([2], len(s))
                continue

            elif is_quantitative(s):
                final_results[pid] = ([2], len(s))
                continue

            elif find_umd(s):
                c = sum(1 for ch in s if ch in CHAR_NUM)
                final_results[pid] = ([-2], c)
                continue

            else:
                t_class, t_cuant = classify_token(s)
                final_results[pid] = ([t_class], t_cuant)
                continue

    return final_results

def normalize_umd_ocr(token: str) -> str:
    """
    Corrección conservadora O/0 para UMD.
    Solo toca el segmento después de '/' y valida con find_umd.
    Ej: C/1O -> C/10, C/O1 -> C/01
    """
    t = token.strip()    
    if not find_umd(t):
        return token

    candidate = re.sub(
        _umd_correct,
        lambda m: m.group(0).replace("O", "0").replace("o", "0"),
        t
    )

    return candidate
