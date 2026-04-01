import re
import logging
import numpy as np
from typing import List, Tuple, Dict, Pattern, Any
from core.utils.math_utils import text_encode
from core.utils.data_utils import CHAR_NUM, ALONE_CHARS

logger = logging.getLogger(__name__)

# Patrón para secuencias especiales de 2 o más caracteres no alfanuméricos (excluyendo espacio, $, ,)
_secuence_pattern: Pattern[str] = re.compile(r'[^a-zA-Z0-9\s$]{2,}')
_sequence_middle_pattern: Pattern[str] = re.compile(r'(?<=[a-zA-Z0-9$])[^a-zA-Z0-9\s$]{2,}(?=[a-zA-Z0-9$])')

_numeric_separator: Pattern[str] =  re.compile(r'^([$\u00A2]?\s*)(-?\d[\d.,]*)(\s*[$\u00A2]?)$', re.IGNORECASE)

_punt_split_pattern: Pattern[str] = re.compile(r'[=;:!?]', re.IGNORECASE)

# Espacios múltiples
_spaces_pattern: Pattern[str] = re.compile(r'\s+')

# Terminación
_termination_pattern: Pattern[str] = re.compile(r'(?i)(s|c|r)?i0n\b', re.IGNORECASE)

# Siglas/Acrónimos
_acronym_pattern: Pattern[str] = re.compile(r'^(?:(?:[A-Za-z]\.){1,}[A-Za-z]\.?|sa|cv|no)(?:[:;,.])?$', re.IGNORECASE)

# RFC
_rfc_code_pattern: Pattern[str] = re.compile(r'^([A-ZÑ&]{3,4})\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])[A-Z0-9]{3}$')
_rfc_acronyms: Pattern[str] = re.compile(r'\b(R\.?F\.?C\.?)\b')

_rfc_code_patterns = re.compile("|".join(p.pattern for p in [_rfc_acronyms, _rfc_code_pattern]))

# IVA
_iva_pattern: Pattern[str] = re.compile(r'\b(I\.?V\.?A\.?)\b')

# Datos Globales
_phone_number: Pattern[str] = re.compile(r'^\d{10}$')
_mail_pattern: Pattern[str] = re.compile(r'^(?:.*@.*|.*mail*|.*\.com)$')
_cp_pattern: Pattern[str] = re.compile(r'^(?:C\.?P\.?\s*)\d{5}$')

_code_patterns: Pattern[str] = re.compile("|".join(p.pattern for p in [_phone_number, _mail_pattern, _cp_pattern]), re.IGNORECASE)

# Fecha
_date_patterns_list: List[Pattern[str]] = [
    re.compile(r'\b(ene(ro)?|feb(rero)?|mar(zo)?|abr(il)?|may(o)?|jun(io)?|jul(io)?|ago(sto)?|sep(t(iembre)?)?|oct(ubre)?|nov(iembre)?|dic(iembre)?)\b'),
    re.compile(r'\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b'),
    re.compile(r'\b(0?[1-9]|[12]\d|3[01])[\/\-\.](0?[1-9]|1[0-2])\b'),
    re.compile(r'\b(199\d|20\d{2})\b'),
    re.compile(r'\b([01]?\d|2[0-3]):[0-5]\d(:[0-5]\d)?\b'),
    re.compile(r'\b([01]\d|2[0-3])[0-5]\d\s*[AaPp]\.?[Mm]\.?\b'),
   # re.compile(r'^\d{8}$')
]

_date_patterns = re.compile("|".join(p.pattern for p in _date_patterns_list), re.IGNORECASE)

# UMD
_umd_correct = re.compile(r'(?<=/)[0-9Oo]+(?=\b)', re.IGNORECASE)

_umd_patterns_list: List[Pattern[str]] = [
    # Masas: kg, g, mg, lb, oz, ton. Incluye variaciones de OCR como kgr.
    re.compile(r'\b\d*([.,]\d+)?\s*(kg(r)?|g(r)?|mg|lb(s)?|oz|ton)\b'),
    # Volúmenes: l, ml, cc, gal. Incluye variaciones como lt, ltr.
    re.compile(r'\b\d*([.,]\d+)?\s*(l(t(r)?)?|ml|cc|gal)\b'),
    # Cantidad: C/ o C/ con número.
    re.compile(r'\b[Cc]\s*/\s*\d*\b'),
    # Longitudes y Áreas: m, cm, mm, km, in, ft, pulg. Detecta la unidad sola o con número.
    # Soporta m2, m^2, m² para áreas.
    re.compile(r'\b(\d+([.,]\d+)?\s*)?(m(t(r)?)?|cm|mm|km|in|ft|pulg|m(t)?(\^2|2|²)|cm(\^2|2|²)|km(\^2|2|²))\b'), 
    # Fracciones (1/2 kg, 1/4) y Dimensiones (10x20).
    re.compile(r'\b(\d+\s*/\s*\d+(\s*(kg(r)?|g(r)?|l(t)?|ml|pz(a)?|ud(s)?))?|[1-9]\s*/\s*(2|4|8|16|32|64)|\d+(?:\s*[xX]\s*\d+)+)\b'),
]

_umd_patterns = re.compile(
    "|".join(p.pattern for p in _umd_patterns_list),
    re.IGNORECASE)

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
_token = re.compile(_token_pattern, re.IGNORECASE)

_start_pattern = rf"^{currency_pattern}\s*{_amount_body_pattern}$"
_start = re.compile(_start_pattern)

_middle_pattern = rf"^{_amount_body_pattern}\s*{currency_pattern}\s*{_amount_body_pattern}$"
_middle = re.compile(_middle_pattern)

_multi_pattern = rf"^(?:\s*{currency_pattern}\s*{_amount_body_pattern}\s*){{2,}}$"
_multi = re.compile(_multi_pattern)

_decimal = re.compile(r"^\d{1,3}(?:[.,]\d{3})*[.,]\d{2,}$")

_quant_runs_patterns = re.compile("|".join(p.pattern for p in [_start, _middle, _multi, _decimal]), re.IGNORECASE)

_end_pattern = rf"^{_amount_body_pattern}\s*{currency_pattern}$"
_end = re.compile(_end_pattern, re.IGNORECASE)

_digits = re.compile(r"\d+")
_split_pattern = rf"{currency_pattern}\s*{_amount_body_pattern}"
_split = re.compile(_split_pattern)

_end_quants = re.compile(r'[.,]00$')

char_num = CHAR_NUM
valid_chars = ALONE_CHARS
char_num_point = char_num.copy()
char_num_point.update(".")

def validate_text(text: str) -> bool:
    if not text or not text.strip():
        return False

    elif not any(char.isalnum() for char in text):
        return False

    if len(text) > 1:
        return True
    else:
        return text in valid_chars
        
def termination_detect(text: str) -> bool:
    if not text:
        return False
    elif len(text) < 6:
        return False
    else:
        return bool(_termination_pattern.search(text))

def is_code(s: str) -> bool:
    if not s:
        return False
    elif len(s) > 17:
        return False
    elif not any(c.isalpha() for c in s):
        return False
    # elif len(s) < 4:
    #     return False
    elif s.isalpha():
        return False
    elif s.isdigit():
        return False
         
    if all(c.isupper() for c in s if c.isalpha()):
        if s.endswith("0"):
            correct: str = s.replace("0", "").strip()
            if correct.isalpha():
                #logger.info(f"No codigo: {s}")
                return False
            elif correct.isdecimal():
                return False
            else:
                return True
        return True
       # return all(c.isupper() for c in s if c.isalpha())
    else:
        return bool(_code_patterns.search(s))

def is_acronym(text: str) -> bool:
    try:
        if not text:
            return False
        return bool(_acronym_pattern.search(text))
    except TypeError as e:
        logger.error(f"Error buscando siglas: {e}", exc_info=True)
    return False

def find_umd(s: str) -> bool:
    try:
        if not s:
            return False
        
        elif len(s) > 11:
            return False
        
        elif not any(c.isalnum() for c in s):
            return False
        
        else:
            return bool(_umd_patterns.search(s))
            
    except TypeError as e:
        logger.warning(f"Error buscando unidades de medida: {e}", exc_info=True)
    return False

def find_date(s: str) -> bool:
    try:
        s = s.replace(" ", "")

        if not s:
            return False
        
        elif s.isalpha():
            return False
        
        elif all(c in char_num_point for c in s):
            return False
        
        is_date = bool(_date_patterns.search(s)) 
        is_umd = bool(_umd_patterns.search(s))
        return is_date

    except TypeError as e:
        logger.error(f"Error buscando fecha: {e}", exc_info=True)
    return False
    
def find_rfc(s: str) -> bool:
    try:
        if not s:
            return False
        
        elif len(s) < 12:
            return False
        
        elif not any(c.isalnum() for c in s):
            return False
        
        else:
            return bool(_rfc_code_patterns.search(s))

    except TypeError as e:
        logger.warning(f"Error buscando RFC: {e}", exc_info=True)
    return False

def find_iva(s: str) -> bool:
    try:
        if not s:
            return False
        
        elif len(s) < 11:
            return False
        
        elif not any(c.isalpha() for c in s):
            return False
        
        return bool(_iva_pattern.search(s))
    except TypeError as e:
        logger.warning(f"Error Buscando IVA: {e}", exc_info=True)
    return False

def contains_quantitative(s: str) -> bool:
    """
    Devuelve True si hay al menos un cuantitativo válido en cualquier parte del texto.
    Retorna False si el texto no contiene ningún dígito.
    """
    if not any(c in CHAR_NUM for c in s):
        return False
    
    runs = find_quantitative_runs(s)
    return len(runs) > 0

def is_quantitative(s_norm: str) -> bool:

  #  s_norm = s.replace("o", "0").replace("O", "0")

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
                possible_num = "".join(ch for ch in maybe_amt if ch in char_num_point)
                if possible_num == "00":
                    return False
                if idx == len(s_norm) - 1:
                    return False
                if idx == 0 or not s_norm[:idx].strip().isdecimal():
                    break

    if _end.match(s_norm):
        return False

    amounts = _digits.findall(s_norm)
    
    if not _end_quants.search(s_norm):
        if any(c == "00" for c in amounts if len(amounts) > 1 or c != "00"):
            return False

    return bool(_quant_runs_patterns.match(s_norm))

def find_quantitative_runs(s: str) -> List[Tuple[int, int, str]]:
    s = (s or "").strip()
    runs: List[Tuple[int, int, str]] = []
    s_norm = s.replace("o", "0").replace("O", "0")

    for m in _token.finditer(s_norm):
        tok = m.group(0)
        if is_quantitative(tok):
            runs.append((m.start(), m.end(), s[m.start():m.end()]))

    currency_count = sum(1 for _, _, tok in runs if currency.search(tok))
    if currency_count > 1:
        split_runs: List[Tuple[int, int, str]] = []
        for match in _split.finditer(s_norm):
            split_runs.append((match.start(), match.end(), s[match.start():match.end()]))
        return split_runs
    return runs

def separate_punt(text: str) -> str:
    if is_acronym(text):
        return text
    else:
        parts = _punt_split_pattern.split(text)
        return " ".join(p.strip() for p in parts if p.strip() and not _punt_split_pattern.fullmatch(p))

def validate_unique_chars(text: str) -> bool:
    """Valida si un caracter unico es válido o no"""
    text = text.strip()
    if len(text) > 1:
       return True
          
    elif text.isdecimal():
        return True
   
    elif text in valid_chars:
        return True

    else:
        return False
    
def space_removal(text: str) -> str:
    """
    Limpia espacios múltiples y espacios iniciales/finales de un texto.
    Reemplaza múltiples espacios consecutivos por un solo espacio y elimina espacios al inicio y final.
    """
    if not text:
        return ""
    
    if text == text.strip() and "  " not in text:
        return text
    # Reemplaza cualquier secuencia de espacios (\s+) por uno solo y limpia bordes
    clean_text = _spaces_pattern.sub(" ", text).strip()    
    return clean_text if clean_text else ""

def remove_special_sequences(text: str) -> str:
    cleaned = _sequence_middle_pattern.sub(' ', text)
    cleaned = _secuence_pattern.sub('', cleaned)
    return cleaned.strip()

def numeric_separator(token: str) -> str:
    """
    Normaliza separadores numéricos en montos cuantitativos.

    Ejemplo:
        3.226.66 -> 3,226.66
    """
    try:
        if not token:
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
                logger.info(f"UMD por no Q: '{s}'")
                return (-2, 0)
            else:
                return (0, 0)
        
        elif all(c in char_num_point for c in s):
            return (2, len(s))
        
        elif contains_quantitative(s):
            return (2, total_cuant)
            
        elif find_umd(s):
            logger.info(f"UMD por función '{s}'")
            return (-2, total_cuant)

        elif is_code(s):
            logger.info(f"Code Por función: {s}")
            return (-1, total_cuant)

        encoded_text = text_encode(s, ["density", "morphological"])
        text_means = np.mean(encoded_text, axis=1, dtype=np.float32)

        if text_means[0] > density_mean[1]:
            return (0, 0)
        
        elif text_means[0] < density_mean[0]:
            if contains_quantitative(s):
                return (2, total_cuant)
            else:
                logger.info(F"CODE por codificación cuant: '{s}'")
                return (1, total_cuant)  # numeric
        
        elif text_means[0] < density_mean[1] and text_means[1] > morph_mean[0]:
            if len(s) < 20:
                logger.info(f"Code por codificación: '{s}'")
                return (-1, total_cuant)  # code
            else:
                return (0, 0)
        else: 
            return (0, 0)  # descriptive

    for pid, polygon in polygons.items():
        s = polygon.ocr_text or ""
        s = s.strip()
        if not s:
            continue

        tokens = s.split()
        total_tokens = len(tokens)
        if not tokens or total_tokens == 0:
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

            elif all(c in char_num_point for c in s) or is_quantitative(s):
                final_results[pid] = ([2], len(s))
                continue

            #elif is_quantitative(s):
              #  final_results[pid] = ([2], len(s))
               # continue

            elif find_umd(s):
                logger.info(f"UMD inicial: '{s}'")
                c = sum(1 for ch in s if ch in CHAR_NUM)
                final_results[pid] = ([-2], c)
                continue
            
            elif is_code(s):
                logger.info(f"CODE INICIAL: '{s}")
                c = sum(1 for ch in s if ch in CHAR_NUM)
                final_results[pid] = ([-1], c)
                continue

            else:
                t_class, t_cuant = classify_token(s)
                final_results[pid] = ([t_class], t_cuant)

    return final_results
