import re
import logging
import unicodedata
from typing import List, Tuple, Dict, Pattern, Set
from core.utils.data_utils import SPECIAL_CHARS, NOT_VALID_CHARS, NOT_VALID_PUNT_CHARS, CHAR_NUM, ALONE_CHARS, PUNC_CHARS

logger = logging.getLogger(__name__)

# Patrón para secuencias especiales de 2 o más caracteres no alfanuméricos (excluyendo espacio, $, ,)
secuence_pattern: Pattern[str] = re.compile(r'[^a-zA-Z0-9\s$]{2,}')
_sequence_middle_pattern: Pattern[str] = re.compile(r'(?<=[a-zA-Z0-9$])[^a-zA-Z0-9\s$]{2,}(?=[a-zA-Z0-9$])')

_numeric_separator: Pattern[str] =  re.compile(r'^([$\u00A2]?\s*)(-?\d[\d.,]*)(\s*[$\u00A2]?)$', re.IGNORECASE)

_punt_split_pattern: Pattern[str] = re.compile(r'([.,;:!?])')
_punt_detect_pattern: Pattern[str] = re.compile(r'[\s\.\-_,;:=]+')

# Espacios múltiples
_spaces_pattern: Pattern[str] = re.compile(r'\s+')

# Terminación
_termination_pattern: Pattern[str] = re.compile(r'(?i)(s|c|r)?i0n\b', re.IGNORECASE)

# Siglas/Acrónimos
_acronym_pattern: Pattern[str] = re.compile(r'^([A-Za-z]\.){2,}[A-Za-z]\.?(?:[:;,])?$', re.IGNORECASE)

# RFC
_rfc_code_pattern: Pattern[str] = re.compile(r'^([A-ZÑ&]{3,4})\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])[A-Z0-9]{3}$', re.IGNORECASE)
_rfc_word_pattern: Pattern[str] = re.compile(r'\b(R\.?F\.?C\.?)\b', re.IGNORECASE)

# IVA
_iva_pattern: Pattern[str] = re.compile(r'\b(I\.?V\.?A\.?)\b', re.IGNORECASE)

# Fechas
_date_patterns: List[Pattern[str]] = [
    re.compile(r'\b(ene(ro)?|feb(rero)?|mar(zo)?|abr(il)?|may(o)?|jun(io)?|ago(sto)?|sep(t(iembre)?)?|oct(ubre)?|nov(iembre)?|dic(iembre)?)\b', re.IGNORECASE),
    re.compile(r'\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}\b'),
    re.compile(r'\b(0?[1-9]|[12]\d|3[01])[\/\-\.](0?[1-9]|1[0-2])\b'),
    re.compile(r'\b(199\d|20\d{2})\b'),
    re.compile(r'\b([01]?\d|2[0-3]):[0-5]\d(:[0-5]\d)?\b'),
    re.compile(r'\b([01]\d|2[0-3])[0-5]\d\s*[AaPp]\.?[Mm]\.?\b'),
]

# UMD
_umd_patterns: List[Pattern[str]] = [
    re.compile(r'\b\d+([.,]\d+)?\s*(kg(r)?|kilo(s)?|g(r|ramo(s)?)?|mg|lb(s)?|libra(s)?|oz|onza(s)?|ton(elada(s)?)?)\b', re.IGNORECASE),
    re.compile(r'\b[Cc]\s*/\s*\d+\b'),
    re.compile(r'\b\d+([.,]\d+)?\s*(l(t(r)?)?|litro(s)?|ml|cc|gal(on(es)?)?)\b', re.IGNORECASE),
    re.compile(r'\b\d+([.,]\d+)?\s*(m(t(r)?)?|metro(s)?|cm|mm|km|in|pulg(ada(s)?)?|ft)\b', re.IGNORECASE),
    re.compile(r'\b\d+([.,]\d+)?\s*(m(t)?(2|\^2|²)|cm(2|\^2|²)|km(2|\^2|²))\b', re.IGNORECASE),
    re.compile(r'\b(m(t)?(2|\^2|²)|cm(2|\^2|²)|km(2|\^2|²))\b', re.IGNORECASE),
    re.compile(r'\b\d+\s*/\s*\d+\s*(kg(r)?|kilo(s)?|g(r)?|l(t(r)?)?|litro(s)?|ml|pz(a)?(s)?|ud(s)?)\b', re.IGNORECASE),
    re.compile(r'\b[1-9]\s*/\s*(2|4|8|16|32|64)\b'),
]

# Cuantitativos
def _build_quantitative_patterns() -> Dict[str, Pattern[str]]:
    digit = r"[0-9oO]"
    currency = r"[$¢]"
    amount_body = rf"(?:{digit}+(?:[.,]{digit}+)?|{digit}{{1,3}}(?:[.,]{digit}{{3}})*)(?:[.,]{digit}+)?"
    token = rf"{currency}\s*{amount_body}|{amount_body}\s*{currency}|{amount_body}"
    return {
        "currency":     re.compile(currency),
        "token":        re.compile(token),
        "start":        re.compile(rf"^{currency}\s*{amount_body}$", re.IGNORECASE),
        "middle":       re.compile(rf"^{amount_body}\s*{currency}\s*{amount_body}$", re.IGNORECASE),
        "end":          re.compile(rf"^{amount_body}\s*{currency}$", re.IGNORECASE),
        "multi":        re.compile(rf"^(?:\s*{currency}\s*{amount_body}\s*){{2,}}$", re.IGNORECASE),
        "decimal":      re.compile(r"^\d{1,3}(?:[.,]\d{3})*[.,]\d{2,}$", re.IGNORECASE),
        "digits":       re.compile(r"\d+"),
        "split":        re.compile(rf"{currency}\s*{amount_body}"),
    }

_Q = _build_quantitative_patterns()

char_num = CHAR_NUM
valid_chars = ALONE_CHARS
special_chars = SPECIAL_CHARS
not_valid_chars = NOT_VALID_CHARS
punc_chars = PUNC_CHARS
not_valid_punt_chars = NOT_VALID_PUNT_CHARS
        
def termination_detect(text: str) -> bool:
    if not validate_text(text):
        return False
    return bool(_termination_pattern.search(text))
    
def find_date(s: str) -> bool:
    try:
        if not s:
            return False
        if any(p.search(s) for p in _date_patterns):
            return True
    except TypeError as e:
        logger.error(f"Error buscando fecha: {e}", exc_info=True)
    return False
    
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
    
def find_rfc(s: str) -> bool:
    try:
        if not s:
            return False
        
        if _rfc_word_pattern.search(s):
            return True
        return bool(_rfc_code_pattern.search(s))
    except Exception as e:
        logger.warning(f"Error buscando RFC: {e}", exc_info=True)
        return False

def find_iva(s: str) -> bool:
    try:
        if not s:
            return False
        
        if _iva_pattern.search(s):
            return True
        return False
    except Exception as e:
        logger.warning(f"Error buscando IVA: {e}", exc_info=True)
    return False
        
def contains_quantitative(s: str) -> bool:
    """
    Devuelve True si hay al menos un cuantitativo válido en cualquier parte del texto.
    """
    runs = find_quantitative_runs(s)
    return len(runs) > 0

def get_quantitative_patterns() -> Dict[str, Pattern[str]]:
    """Devuelve los patrones cuantitativos ya compilados."""
    return _Q

def find_quantitative(s: str) -> bool:
    s = (s or "").strip()
    if not validate_text(s) or "%" in s:
        return False

    s_norm = s.replace("o", "0").replace("O", "0")

    currency_symbols = "$¢"
    for sym in currency_symbols:
        idx = s_norm.find(sym)
        if idx != -1:
            after = s_norm[idx+1:]
            if any(c.isdigit() for c in after):
                maybe_amt = after.lstrip()
                possible_num = "".join(ch for ch in maybe_amt if ch.isdigit() or ch in ".,")
                if possible_num == "00":
                    return False
                if idx == len(s_norm) - 1:
                    return False
                if idx == 0 or not s_norm[:idx].strip().isdigit():
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
    s = (s or "").strip()
    
    runs: List[Tuple[int, int, str]] = []
    s_norm = s.replace("o", "0").replace("O", "0")

    for m in _Q["token"].finditer(s_norm):
        tok = m.group(0)
        if find_quantitative(tok):
            runs.append((m.start(), m.end(), s[m.start():m.end()]))

    currency_count = sum(1 for _, _, tok in runs if _Q["currency"].search(tok))
    if currency_count > 1:
        split_runs: List[Tuple[int, int, str]] = []
        for match in _Q["split"].finditer(s_norm):
            split_runs.append((match.start(), match.end(), s[match.start():match.end()]))
        return split_runs

    return runs

def separate_punt(text: str) -> List[str]:
    return _punt_split_pattern.split(text)

def detect_punt(text: str) -> bool:
    return _punt_detect_pattern.fullmatch(text) is not None

def remove_special_sequences(text: str) -> str:    
    cleaned = _sequence_middle_pattern.sub(' ', text)
    cleaned = secuence_pattern.sub('', cleaned)
    return cleaned.strip()

def is_special_sequence(text: str) -> bool:
    """
    Devuelve True si el texto completo consiste ÚNICAMENTE en una secuencia 
    de 2 o más caracteres especiales consecutivos (ruido de OCR).
    """
    # Usamos fullmatch para asegurar que TODO el string sea la secuencia de ruido
    # Reutilizamos tu lógica: no alfanuméricos excluyendo espacio y $
    # {2,} asegura que sean 2 o más.
    return secuence_pattern.fullmatch(text.strip()) is not None

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
    text = text.strip().lower()
    if len(text) > 1:
       return True

    if text.isnumeric():
        return True
    
    elif text in valid_chars:
        return True

    else:
        return False
        
def norm_text(text: str) -> str:
    if not validate_text(text):
        return ""
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8').lower()

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

def text_compacter(text: str) -> str:
    """
    Elimina TODOS los espacios del texto para vectorización.
    Mantiene mayúsculas, minúsculas y acentos originales.
    """
    if not validate_text(text):
        return ""
    
    # Usamos el patrón compilado para eliminar espacios eficientemente
    return _spaces_pattern.sub("", text)

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
    
    # Buscamos al menos un caracter que sea letra o número
    # Cualquier cosa que no tenga letras (incluidas con acento) ni números es ruido
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
        if not find_quantitative(t):
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
        if not (frac.isdigit() and len(frac) == 2):
            return token

        int_digits = re.sub(r"[.,]", "", number[:last_sep])
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