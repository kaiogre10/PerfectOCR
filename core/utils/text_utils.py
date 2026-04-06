import re
import logging
from typing import List, Tuple, Dict, Pattern, Any
from core.utils.math_utils import text_encode
from core.utils.data_utils import CHAR_NUM, ALONE_CHARS, VALID_NUM_PUNT_CHARS

logger = logging.getLogger(__name__)

_base_zeros_pattern: Pattern[str] = re.compile(r'^[0O]{2}$')
_base_date_num_str = r'[0123O][0-9O]' # Bloque base para días/meses (máx 3X o OX)
_base_date_num: Pattern[str] = re.compile(rf'^{_base_date_num_str}$')

# Patrón para secuencias especiales de 2 o más caracteres no alfanuméricos (excluyendo espacio, $, ,)
_secuence_pattern: Pattern[str] = re.compile(r'[^a-zA-Z0-9\s$]{2,}')
_sequence_middle_pattern: Pattern[str] = re.compile(r'(?<=[a-zA-Z0-9$])[^a-zA-Z0-9\s$]{2,}(?=[a-zA-Z0-9$])')

# _numeric_separator: Pattern[str] =  re.compile(r'^([$\u00A2]?\s*)(-?\d[\d.,]*)(\s*[$\u00A2]?)$', re.IGNORECASE)
_hour_pattern: Pattern[str] = re.compile(rf'\b{_base_date_num_str}:[0-5O][0-9O](?::[0-5O][0-9O])?\b')
_punt_split_pattern: Pattern[str] = re.compile(r'[=;:!?]', re.IGNORECASE)

# Espacios múltiples
_spaces_pattern: Pattern[str] = re.compile(r'\s+')

# Terminación
_termination_pattern: Pattern[str] = re.compile(r'(?i)(s|c|r)?i0n\b', re.IGNORECASE)

# Siglas/Acrónimos
_acronym_pattern: Pattern[str] = re.compile(r'^(?:(?:[A-Za-z]\.){1,}[A-Za-z]\.?|sa|cv|no)(?:[:;,.])?$', re.IGNORECASE)

# Datos Globales
_numeric_code: Pattern[str] = re.compile(r'^[0O]\d+$')
_phone_number: Pattern[str] = re.compile(r'^\d{10}$')
_mail_pattern: Pattern[str] = re.compile(r'^(?:.*@.*|.*mail*|.*\.com)$')
_cp_pattern: Pattern[str] = re.compile(r'^(?:C\.?P\.?\s*)\d{5}$')

_code_patterns: Pattern[str] = re.compile("|".join(p.pattern for p in [_phone_number, _mail_pattern, _cp_pattern]), re.IGNORECASE)

# Fecha
_date_patterns_list: List[Pattern[str]] = [
    # Fechas completas y día/mes: Usa el bloque base para evitar confundirse con fracciones/cuantitativos
    re.compile(rf'\b{_base_date_num_str}[\s\/\-]{_base_date_num_str}(?:[\s\/\-](?:\d{{2,4}}))?\b'),
    # Meses (palabras)
    re.compile(r'\b(ene(ro)?|feb(rero)?|mar(zo)?|abr(il)?|may(o)?|jun(io)?|jul(io)?|ago(sto)?|sep(t(iembre)?)?|oct(ubre)?|nov(iembre)?|dic(iembre)?)\b'),
    # Años
    re.compile(r'\b(199\d|20\d{2})\b'),
]

_umd_patterns_list: List[Pattern[str]] = [
    # Masas: kg, g, mg, lb, oz, ton. Incluye variaciones de OCR como kgr.
    re.compile(r'\b\d*([.,]\d+)?\s*(kg?|g(r)?(s)?|mg|lb(s)?|oz|ton)\b'),
    # Volúmenes: l, ml, cc, gal. Incluye variaciones como lt, ltr.
    re.compile(r'\b\d*([.,]\d+)?\s*(l(t(r)?(s)?)?|ltrs?|lts?|ml|cc|gal)\b'),
    # Cantidad: C/ o C/ con número.
    re.compile(r'\b[Cc]\s*/\s*\d*\b'),
    # Longitudes y Áreas: m, cm, mm, km, in, ft, pulg. Detecta la unidad sola o con número. Soporta m2, m^2, m² para áreas.
    re.compile(r'\b(\d+([.,]\d+)?\s*)?(m(t(s)?)?|cm|mm|km|in|ft|pul|m(t)?(\^2|2|²)|cm(\^2|2|²)|km(\^2|2|²))\b'),
    re.compile(r'\b[1-9]\d{0,2}\s*/\s*[1-9]\d{0,2}\b'),
    # Fracciones (1/2 kg, 1/4).
  #  re.compile(r'\b\d+\s*/\s*\d+\b'),
    # Dimensiones (10x20)
    re.compile(r'\b\d+(?:\s*[xX]\s*\d+)+\b')
]

_umd_patterns = re.compile("|".join(p.pattern for p in _umd_patterns_list), re.IGNORECASE)

# Define los patrones como strings
digit_pattern = r"[0-9oO]"
currency_pattern = r"[$]"

# Compila los patrones base
currency = re.compile(currency_pattern)

# Usa los strings en las interpolaciones
_amount_body_pattern = (
    rf"(?:{digit_pattern}+(?:[.,]{digit_pattern}+)?|" # Caso simple: 10.50
    rf"{digit_pattern}{{1,3}}(?:[.,]{digit_pattern}{{3}})*)(?:[.,]{digit_pattern}{{2}})" # Caso miles: 1,000.00
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
_end_quants = re.compile(r'[.,]00$')

_split_pattern = rf"{currency_pattern}\s*{_amount_body_pattern}"
_split = re.compile(_split_pattern)

RFC_PATTERNS: Pattern[str] = re.compile(r'^([A-ZÑ&]{3,4})\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])[A-Z0-9]{3}$')
# _rfc_acronyms: Pattern[str] = re.compile(r'\b(R\.?F\.?C\.?)\b')

# RFC_PATTERNS = re.compile("|".join(p.pattern for p in [_rfc_code_pattern, _rfc_acronyms]))
IVA_PATTERN: Pattern[str] = re.compile(r'\b(I\.?V\.?A\.?)\b')
DATE_PATTENRS = re.compile("|".join(p.pattern for p in _date_patterns_list), re.IGNORECASE)
valid_chars = ALONE_CHARS
char_num_point = CHAR_NUM.copy()
char_num_point.update(".")

def validate_text(text: str) -> bool :
    """valida que un string contenga caracteres válidos y que no esté vacío"""
    total_txt = len(text.strip())
    # Si tiene más de un carácter, debe tener al menos un alfanumérico
    if total_txt > 1:
        return any(char.isalnum() for char in text)
    # Si es un solo carácter, debe ser válido (número o en valid_chars)
    elif total_txt == 1:
        return text in valid_chars or text.isdecimal()
    else:
        return False
         
def validate_unique_chars(text: str) -> bool:
    """Valida si un caracter unico es válido o no"""
    text = text.strip()
    if len(text) != 1:
        return False
    else:
        return text in valid_chars or text.isdecimal()

def termination_detect(text: str) -> bool:
    if not text:
        return False
    elif len(text) < 6:
        return False
    else:
        return bool(_termination_pattern.search(text))

def is_code(s: str) -> bool:
    if not any(c.isalpha() for c in s):
        return False
    elif s.isalpha():
        return False
    elif _numeric_code.search(s) and not contains_quantitative(s):
        logger.info(f"Código comienza n 0")
        return True
    else:
        return bool(_code_patterns.search(s))

def is_acronym(text: str) -> bool:
    try:
        if not text:
            return False
        return bool(_acronym_pattern.search(text))
    except TypeError as e:
        logger.warning(f"Error buscando siglas: {e}", exc_info=True)
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
    
def is_quantitative(text: str) -> bool:
    """
    1. VALIDACIÓN ATÓMICA: 
    No corrige, no hace strip, no filtra. Evalúa estrictamente 
    si un string completo coincide con el formato cuantitativo.
    """
    if not text:
        return False
    
    return all(c in char_num_point for c in text) or bool(_quant_runs_patterns.match(text))

def contains_quantitative(text: str) -> bool:
    """
    2. ESCÁNER:
    Recibe un string completo. Escanea el texto y devuelve True 
    si encuentra algún sub-string que sea puramente cuantitativo.
    """
    if not text or text.isalpha():
        return False
    
    for match in _token.finditer(text):
        if is_quantitative(match.group(0)):
            return True
            
    return False

def get_cuants(text: str) -> str:
    """
    3. EXTRACTOR / SEPARADOR:
    Recibe un string, detecta cuantitativos incrustados en ruido y 
    los aísla con espacios. Nunca devuelve None.
    Ej: '879.00$3.67X' -> '879.00 $3.67 X'
    """
    if not text:
        return ""
   
    elif not contains_quantitative(text):
        return text

    matches = list(_token.finditer(text))
    if not matches:
        return text

    result = text
    # Se itera en reversa para no alterar los índices al inyectar los espacios
    for m in reversed(matches):
        tok = m.group(0)
        if is_quantitative(tok):
            start, end = m.span()
            # Aísla el cuantitativo inyectando espacios a sus lados
            result = result[:start] + f" {tok} " + result[end:]

    # Limpia los espacios múltiples generados por la inyección
    # logger.info(f"Text: '{text}' -> Cuants: '{result}'")
    return result

def separate_punt(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    
    if text.isalnum():
        return text

    if is_acronym(text):
        return text
   
    if is_quantitative(text):
        return text

    tokens = text.split()
    processed_tokens: List[str] = []
    for t in tokens:
        if t.endswith("."):
            t = t.replace(".", "")
        # Si el token actual NO es una hora, sepárale la puntuación.
        if not _hour_pattern.fullmatch(t):
            # Reemplaza los signos de puntuación por un espacio en este token.
            cleaned_token = _punt_split_pattern.sub(' ', t)
            processed_tokens.append(cleaned_token)
        else:
            # Si el token ES una hora, déjalo como está.
            processed_tokens.append(t)
    
    # Une los tokens procesados y limpia los espacios extra.
    return " ".join(processed_tokens)

def normalice_text(text: str) -> str:
    """Normaliza eliminando ruido, no aplica formato o codificación"""
    if not text.strip():
        return ""
   
    elif not contains_quantitative(text):
        punt_text = separate_punt(text)
        if not punt_text:
            return ""
        return remove_special_sequences(punt_text)
    else:
        return space_removal(text)

def space_removal(text: str) -> str:
    """
    Limpia espacios múltiples y espacios iniciales/finales de un texto.
    Reemplaza múltiples espacios consecutivos por un solo espacio y elimina espacios al inicio y final.
    """
    if not text:
        return ""
    
    if " " not in text and text == text.strip():
        return text
    # Reemplaza cualquier secuencia de espacios (\s+) p%\bor uno solo y limpia bordes
    clean_text = _spaces_pattern.sub(" ", text).strip()
    return clean_text if clean_text else text.strip()

def remove_special_sequences(text: str) -> str:
    """
    Elimina secuencias especiales de dos o más caracteres no alfanuméricos.
    Conserva los caracteres sueltos válidos, pero reemplaza por un espacio las
    secuencias internas de símbolos, y luego limpia espacios sobrantes.
    Ejemplo:
        remove_special_sequences("abc@@def!!ghi") -> 'abc def ghi'
    """
    cleaned = _sequence_middle_pattern.sub(' ', text)
    cleaned = _secuence_pattern.sub('', cleaned)
    return cleaned if cleaned else ""
        
def clasify_words(polygons: Dict[str, Any], worker_config: Dict[str, Any] ) -> Dict[str, Tuple[List[int], int]]:
    density_thr: Tuple[float, float] = worker_config["encode_mean"]
    morph_thr: Tuple[float, float] = worker_config["morph_mean"]
    final_results: Dict[str, Tuple[List[int], int]] = {}

    def classify_token(s: str) -> Tuple[int, int]:
        if not s:
            return (0, 0)
        total_text = len(s)
        if s.isalpha():
            return (0, 0)
    
        elif s.isdecimal():
            return (1, total_text)

        total_cuant = sum(1 for ch in s if ch in CHAR_NUM)

        if total_cuant == 0:
            if find_umd(s):
                return (-2, 0)
            else:
                # logger.info(f"DESCRIP. no UMD: '{s}'")
                return (0, 0)
        
        elif contains_quantitative(s):
            return (2, total_cuant)
            
        elif find_umd(s):
            #logger.info(f"UMD por función '{s}'")
            return (-2, total_cuant)

        elif is_code(s):
            #logger.info(f"Code Por función: {s}")
            return (-1, total_cuant)

        encoders = text_encode(s.lower(), ["all"])
        
        dense_mean = float(sum(encoders[0]) / total_text)
        morphology_mean = float(sum(encoders[1]) / total_text)
        # frec_mean = float(sum(encoders[2])) / total_text
        # logger.info(f"CODIFICACIONES PARA: '{s}':"
        #             "\n"f"{encoders}"
        #             "\n"f"{dense_mean, morphology_mean, frec_mean}")

        if dense_mean > density_thr[1]:
            return (0, 0)
        
        elif dense_mean < density_thr[0]:
            if s.isnumeric() or any(c in VALID_NUM_PUNT_CHARS for c in s):
                # logger.info(F"NUME por codificación cuant: '{s}'")
                return (1, total_cuant)
            else:
                # logger.info(F"CODE por codificación cuant: '{s}'")
                return (-1, total_cuant)  # numeric
        
        elif dense_mean < density_thr[1] and morphology_mean > morph_thr[0]:
                #logger.info(f"Code por codificación: '{s}'")
            return (-1, total_cuant)  # code
            # else:
            #     return (0, 0)
        else: 
            return (0, 0)  # descriptive

    for pid, polygon in polygons.items():
        s = polygon.ocr_text or ""
        s = s.strip()
        if not s:
            continue
       
        # s = get_cuants(s)

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

            elif is_quantitative(s):
                final_results[pid] = ([2], len(s))
                # logger.info(f"CUANT POR SET: '{s}'")
                continue

            elif find_umd(s):
                #logger.info(f"UMD inicial: '{s}'")
                c = sum(1 for ch in s if ch in CHAR_NUM)
                final_results[pid] = ([-2], c)
                continue
            
            elif is_code(s):
                #logger.info(f"CODE INICIAL: '{s}")
                c = sum(1 for ch in s if ch in CHAR_NUM)
                final_results[pid] = ([-1], c)
                continue

            else:
                # logger.info(f"Token sin clasificación: '{s}'")
                t_class, t_cuant = classify_token(s)
                final_results[pid] = ([t_class], t_cuant)
                continue

    return final_results
