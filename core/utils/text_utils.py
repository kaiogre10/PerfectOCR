import re
import logging
from typing import List, Tuple, Dict, Pattern, Any, Set, Optional, Set
from core.utils.math_utils import text_encode
from core.utils.data_utils import CHAR_NUM, ALONE_CHARS, VALID_NUM_PUNT_CHARS

logger = logging.getLogger(__name__)

_zeros_str = r'[O0QDo]'
_base_date_num_str = r'[0123O][0-9O]'
_zeros_pattern = re.compile(_zeros_str, re.IGNORECASE)

# Patrón para secuencias especiales de 2 o más caracteres no alfanuméricos (excluyendo espacio, $, /,)
_secuence_pattern: Pattern[str] = re.compile(r'[^a-zA-Z0-9\s/$]{2,}', re.IGNORECASE)
_sequence_middle_pattern: Pattern[str] = re.compile(r'(?<=[a-zA-Z0-9$])[^a-zA-Z0-9\s$]{2,}(?=[a-zA-Z0-9$])', re.IGNORECASE)

_hour_pattern: Pattern[str] = re.compile(rf'\b{_base_date_num_str}:[0-5O][0-9O](?::[0-5O][0-9O])?\b', re.IGNORECASE)
_punt_split_pattern: Pattern[str] = re.compile(r"[*_'=.,:;&-]")
_edge_punt_pattern = re.compile(rf'^({_punt_split_pattern.pattern}+)|({_punt_split_pattern.pattern}+)$', re.IGNORECASE)

# Espacios múltiples
_spaces_pattern: Pattern[str] = re.compile(r'\s+')
_spaces_pattern: Pattern[str] = re.compile(r'\s+')

# Siglas/Acrónimos
_acronym_pattern: Pattern[str] = re.compile(r'^(?:(?:[A-Za-z]\.){1,}[A-Za-z]\.?|sa|cv)(?:[:;,.])?$', re.IGNORECASE)

# Datos Globales
_phone_number: Pattern[str] = re.compile(r'^\d{10}$')
_mail_pattern: Pattern[str] = re.compile(r'.*@.+', re.IGNORECASE)
_cp_pattern: Pattern[str] = re.compile(r'^(?:C\.?P\.?\s*)\d{5}$', re.IGNORECASE)

_numeric_code: Pattern[str] = re.compile(rf'^{_zeros_str}[0-9]+$')

_code_patterns: Pattern[str] = re.compile("|".join(p.pattern for p in [_phone_number, _mail_pattern, _cp_pattern]), re.IGNORECASE)

# Fecha
_date_patterns_list: List[Pattern[str]] = [
    # Día + mes en letras + año en un solo string OCR (ej. "21 mar 2023")
    re.compile(rf'\b{_base_date_num_str}\s+(?:ene(?:ro)?|feb(?:rero)?|mar(?:zo)?|abr(?:il)?|may(?:o)?|jun(?:io)?|jul(?:io)?|ago(?:s(?:to)?)?|sep(?:t(?:iembre)?)?|oct(?:ubre)?|nov(?:iembre)?|dic(?:iembre)?)\s+(?:19\d{{2}}|20\d{{2}})\b', re.IGNORECASE),
    # Fechas completas y día/mes: Usa el bloque base para evitar confundirse con fracciones/cuantitativos
    re.compile(rf'\b{_base_date_num_str}[\s\/\-]{_base_date_num_str}(?:[\s\/\-](?:\d{{2,4}}))?\b', re.IGNORECASE),
    # Meses (palabras)
    re.compile(r'\b(ene(ro)?|feb(rero)?|mar(zo)?|abr(il)?|may(o)?|jun(io)?|jul(io)?|ago(s(to)?)?|sep(t(iembre)?)?|oct(ubre)?|nov(iembre)?|dic(iembre)?)\b', re.IGNORECASE),
    # Años
    re.compile(r'\b(199\d|20\d{2})\b', re.IGNORECASE)
]

_mass_str = r'(kg?|g(r)?(s)?|mg|lb(s)?|oz|ton)\b'
_vol_str = r'(l(t(r)?(s)?)?|ltrs?|lts?|ml|cc|gal)\b'

_umd_patterns_list: List[Pattern[str]] = [
    # Masas: kg, g, mg, lb, oz, ton. Incluye variaciones de OCR como kgr.
    re.compile(rf'\b\d*([.,]\d+)?\s*{_mass_str}', re.IGNORECASE),
    # Volúmenes: l, ml, cc, gal. Incluye variaciones como lt, ltr.
    re.compile(rf'\b\d*([.,]\d+)?\s*{_vol_str}', re.IGNORECASE),
    # Cantidad: C/ o C/ con número.
    re.compile(r'\b[Cc]\s*/\s*\d*\b', re.IGNORECASE),
    re.compile(r'\b[Cc]\s*/\s*\d*\b', re.IGNORECASE),
    # Longitudes y Áreas: m, cm, mm, km, in, ft, pulg. Detecta la unidad sola o con número. Soporta m2, m^2, m² para áreas.
    re.compile(r'\b(\d+([.,]\d+)?\s*)?(m(t(s)?)?|cm|mm|km|in|ft|m(t)?(\^2|2|²)|cm(\^2|2|²)|km(\^2|2|²))\b', re.IGNORECASE),

    re.compile(r'\b(\d+([.,]\d+)?\s*)?(m(t(s)?)?|cm|mm|km|in|ft|m(t)?(\^2|2|²)|cm(\^2|2|²)|km(\^2|2|²))\b', re.IGNORECASE),

    re.compile(r'\b[1-9]\d{0,2}\s*/\s*[1-9]\d{0,2}\b'),
    # Fracciones (1/2 kg, 1/4).
  #  re.compile(r'\b\d+\s*/\s*\d+\b'),
    # Dimensiones (10x20)
    re.compile(r'\b\d+(?:\s*[xX]\s*\d+)+\b')
]

_len_pattern = re.compile(r'(m(t(s)?)?|cm|mm|km|in|ft)\b', re.IGNORECASE)
_mass_pattern = re.compile(_mass_str, re.IGNORECASE)
_vol_pattern = re.compile( _vol_str,re.IGNORECASE)

_size_pattern = re.compile(r'\b(gde|med|ch|paq)\b', re.IGNORECASE)
_mesure_patterns = re.compile("|".join(p.pattern for p in [_size_pattern, _mass_pattern, _vol_pattern, _len_pattern]), re.IGNORECASE)

_umd_patterns = re.compile("|".join(p.pattern for p in _umd_patterns_list), re.IGNORECASE)

# Define los patrones como strings
digit_pattern = r"[0-9oOQ]"
digit_pattern = r"[0-9oOQ]"
currency_pattern = r"[$]"
_clean_currency_pattern = (rf'\b[{currency_pattern},]')
# Patrón: S al inicio, al menos 3 dígitos entre la S y un punto o coma
# _s_correct_pattern = re.compile(r'^S\d{3,}[.,]', re.IGNORECASE)

_clean_currency_pattern = (rf'\b[{currency_pattern},]')
# Patrón: S al inicio, al menos 3 dígitos entre la S y un punto o coma
# _s_correct_pattern = re.compile(r'^S\d{3,}[.,]', re.IGNORECASE)

# _currency_stick_pattern: Pattern[str] = re.compile(r'([a-zA-Z])([\$])')

# Compila los patrones base
_clean_currency = re.compile(_clean_currency_pattern, re.IGNORECASE)
_clean_currency = re.compile(_clean_currency_pattern, re.IGNORECASE)

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
# Detecta patrones cuantitativos en texto:
# Detecta patrones cuantitativos en texto:
_token = re.compile(_token_pattern)

# Patrón: Montos con símbolo al inicio ($ 80.50)
# Patrón: Montos con símbolo al inicio ($ 80.50)
_start_pattern = rf"^{currency_pattern}\s*{_amount_body_pattern}$"
_start = re.compile(_start_pattern)

# Patrón: Monto con símbolo en medio (80 $ 50)
# Patrón: Monto con símbolo en medio (80 $ 50)
_middle_pattern = rf"^{_amount_body_pattern}\s*{currency_pattern}\s*{_amount_body_pattern}$"
_middle = re.compile(_middle_pattern)

# Patrón: Múltiples montos seguidos de símbolo ($100 $200)
# Patrón: Múltiples montos seguidos de símbolo ($100 $200)
_multi_pattern = rf"^(?:\s*{currency_pattern}\s*{_amount_body_pattern}\s*){{2,}}$"
_multi = re.compile(_multi_pattern)

# Patrón: Decimales grandes tipo 1,230.50 (sin $)
# Patrón: Decimales grandes tipo 1,230.50 (sin $)
_decimal = re.compile(r"^\d{1,3}(?:[.,]\d{3})*[.,]\d{2,}$")

_quant_runs_patterns = re.compile("|".join(p.pattern for p in [_decimal, _start, _middle, _multi]), re.IGNORECASE)
_quant_runs_patterns = re.compile("|".join(p.pattern for p in [_decimal, _start, _middle, _multi]), re.IGNORECASE)

# Patrón: Monto terminado en símbolo (80.00 $)
# _end_pattern = rf"^{_amount_body_pattern}\s*{currency_pattern}$"
# _end = re.compile(_end_pattern, re.IGNORECASE)
# Patrón: Monto terminado en símbolo (80.00 $)
# _end_pattern = rf"^{_amount_body_pattern}\s*{currency_pattern}$"
# _end = re.compile(_end_pattern, re.IGNORECASE)

# Extrae solo los dígitos (sin formato decimal)
# _digits = re.compile(r"\d+")

# Detecta terminaciones típicas de dinero (.00 ó ,00)
# _end_quants = re.compile(r'[.,]00$', re.IGNORECASE)
# Extrae solo los dígitos (sin formato decimal)
# _digits = re.compile(r"\d+")

# Detecta terminaciones típicas de dinero (.00 ó ,00)
# _end_quants = re.compile(r'[.,]00$', re.IGNORECASE)

# Patrón equivalente a _split, pero requiere $ al inicio y una cantidad
# Patrón equivalente a _split, pero requiere $ al inicio y una cantidad
_split_pattern = rf"{currency_pattern}\s*{_amount_body_pattern}"
_split = re.compile(_split_pattern, re.IGNORECASE)

_rfc_acronyms: Pattern[str] = re.compile(r'\b(R\.?F\.?C\.?)\b', re.IGNORECASE)
_rfc_pattern: Pattern[str] = re.compile(r'^([A-ZÑ]{3,4})\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])[A-Z0-9]{3}$', re.IGNORECASE)
_rfc_patterns: Pattern[str] = re.compile("|".join(p.pattern for p in [_rfc_pattern, _rfc_acronyms]), re.IGNORECASE)

_iva_pattern: Pattern[str] = re.compile(r'\b(I\.?V\.?A\.?)\b', re.IGNORECASE)
_date_patterns = re.compile("|".join(p.pattern for p in _date_patterns_list), re.IGNORECASE)

char_num_point: Set[str] = CHAR_NUM.copy()
char_num_point.update(".")

def validate_text(text: str) -> bool :
    """valida que un string contenga caracteres válidos y que no esté vacío"""
    text = text.strip()
    total_txt = len(text)
    # Si tiene más de un carácter, debe tener al menos un alfanumérico
    if total_txt > 1:
        return any(char.isalnum() for char in text)
    # Si es un solo carácter, debe ser válido (número o en ALONE_CHARS)
    if total_txt == 1:
        return text.isdecimal() or text in ALONE_CHARS
    return False

def is_code(s: str) -> bool:
    if not any(c.isalpha() for c in s):
        return False
    if s.isalpha():
        return False
    if bool(_numeric_code.match(s)):
        logger.info(f"Código comienza n 0")
        return True
    return bool(_code_patterns.search(s))

def find_key_data(s: str, activate_func: List[bool]) -> Optional[int]:
    """
    Busca fecha (9), RFC (7) o IVA (8) en el texto crudo del polígono.
    activate_func: [fecha_ya_encontrada, rfc_ya_encontrado, iva_ya_encontrado];
    se pone True en el índice correspondiente al devolver un key_field distinto de 0.
    Prioridad: fecha > RFC > IVA (solo un tipo por llamada).
    """
    try:
        s = s.strip()
        if not any(c.isalnum() for c in s):
            return None

        if not activate_func[0] and bool(_date_patterns.search(s)):
            activate_func[0] = True
            return 9

        if not activate_func[1] and bool(_rfc_patterns.search(s)):
            activate_func[1] = True
            return 7

        if not activate_func[2] and bool(_iva_pattern.search(s)):
            activate_func[2] = True
            return 8

        return None

    except ValueError as e:
        logger.warning(f"Error buscando datos globales: {e}", exc_info=True)
    return None

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
        if not any(c.isalnum() for c in s):
            return False
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
    if not text or len(text) < 3:
        return False
    
    return all(c in char_num_point for c in text) or bool(_quant_runs_patterns.fullmatch(text))

def contains_quantitative(text: str) -> bool:
    """
    2. ESCÁNER:
    Devuelve True si encuentra algún sub-string cuantitativo en el texto.
    """
    if not text or text.isalpha():
        return False

    match = _token.search(text)
    return bool(match and is_quantitative(match.group(0)))

def get_cuants(text: str) -> str:
    """
    3. EXTRACTOR / SEPARADOR:
    Aísla cuantitativos SÓLO si están pegados a otros caracteres (ruido o texto).
    Si ya están separados por espacios, no modifica el texto.
    """
    text = text.strip()
    if not text:
        return ""
        
    if text.isalpha() or len(text) < 3:
        return text

    words = text.split(" ")
    result_parts: List[str] = []
    
    for word in words:
        if is_quantitative(word) and word.count("$") >= 2:
            compact = word.replace(" ", "")
            chunks = [m.group(0).replace(" ", "") for m in _split.finditer(compact)]
            if len(chunks) >= 2 and "".join(chunks) == compact:
                result_parts.append(" ".join(chunks))
                continue

        if is_quantitative(word) and word.count("$") == 1:
            result_parts.append(word)
            continue

        matches = list(_token.finditer(word))
        if not matches:
            result_parts.append(word)
            continue
        
        result = word
        for m in reversed(matches):
            tok = m.group(0)
            start, end = m.span()

            # Caso clave: cuantitativo válido pegado a letras (ej. "93v", "v93")
            if is_quantitative(tok):
                needs_left_space = start > 0 and result[start - 1].isalpha()
                needs_right_space = end < len(result) and result[end].isalpha()

                if needs_left_space or needs_right_space:
                    left_part = result[:start]
                    right_part = result[end:]
                    mid = f"{' ' if needs_left_space else ''}{tok}{' ' if needs_right_space else ''}"
                    result = left_part + mid + right_part


            # Caso clave: cuantitativo válido pegado a letras (ej. "93v", "v93")
            if is_quantitative(tok):
                needs_left_space = start > 0 and result[start - 1].isalpha()
                needs_right_space = end < len(result) and result[end].isalpha()

                if needs_left_space or needs_right_space:
                    left_part = result[:start]
                    right_part = result[end:]
                    mid = f"{' ' if needs_left_space else ''}{tok}{' ' if needs_right_space else ''}"
                    result = left_part + mid + right_part

        result_parts.append(result)
    
    return " ".join(result_parts).strip()
    # return _zeros_pattern.sub("0", quants).strip()

def clean_cuant(text: str) -> str:
    """Normaliza texto para Decimal"""
    text_0 = _zeros_pattern.sub("0", text).strip()
    return _clean_currency.sub('', text_0).strip()

def clean_cuant(text: str) -> str:
    """Normaliza texto para Decimal"""
    text_0 = _zeros_pattern.sub("0", text).strip()
    return _clean_currency.sub('', text_0).strip()

def clean_punct(text: str) -> str:
    """
    Elimina los caracteres de puntuación definidos en _punt_split_pattern
    que se encuentran al inicio y al final de cada token en el texto.
    """
    text = text.strip()
    if not text:
        return ""
        
    if is_acronym(text):
        logger.info(f"Acromimo: {text}")
        return text

    tokens = text.split()
    processed_tokens: List[str] = []
    # Patrón para encontrar puntuación al inicio o al final de un token
    for t in tokens:
        # Elimina la puntuación del inicio y del final del token
        cleaned_token = _edge_punt_pattern.sub('', t).strip()
        processed_tokens.append(cleaned_token)

    return " ".join(processed_tokens).strip()

def separate_punt(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    
    if is_acronym(text):
        logger.info(f"Acromimo: {text}")
        return text
   
    processed_tokens: List[str] = []
    tokens = text.split()
    for t in tokens:
        t = t.strip()
        # Mantiene intactas horas y cuantitativos puros; limpia los tokens mixtos.
        if not bool(_hour_pattern.fullmatch(t)) and not is_quantitative(t):
            cleaned_token = _punt_split_pattern.sub(" ", t)
            processed_tokens.append(cleaned_token)
        else:
            # Si es una hora, se mantiene intacta
            processed_tokens.append(t)

    # Une los tokens y usa space_removal para normalizar todos los espacios
    return " ".join(processed_tokens).strip()

def space_removal(text: str) -> str:
    """
    Limpia espacios múltiples y espacios iniciales/finales de un texto.
    Reemplaza múltiples espacios consecutivos por un solo espacio y elimina espacios al inicio y final.
    """
    if not text:
        return ""
    
    if " " not in text or text == text.strip():
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
    cleaned = _sequence_middle_pattern.sub(" ", text).strip()
    cleaned = _secuence_pattern.sub("", cleaned)
    return cleaned if cleaned else ""
        
def clasify_words(polygons: Dict[str, Any], worker_config: Dict[str, Any] ) -> Dict[str, Tuple[List[int], int]]:
    density_thr: Tuple[float, float] = worker_config["encode_mean"]
    morph_thr: Tuple[float, float] = worker_config["morph_mean"]
    final_results: Dict[str, Tuple[List[int], int]] = {}

    def classify_token(s: str) -> Tuple[int, int]:
        if not s:
            return (0, 0)
            
        total_text = len(s)
        total_cuant = sum(1 for ch in s if ch in CHAR_NUM)
        
        if s.isalpha() or total_cuant == 0:
            if bool(_mesure_patterns.search(s)):
                return (-2, 0)
            return (0, 0)
    
        if s.isdecimal():
            if s.startswith("0"):
                # logger.info(f"Code por inicio 0: {s}")
                return (-1, total_text)
            return (1, total_text)
        
        if contains_quantitative(s):
            return (2, total_cuant)
            
        if find_umd(s):
            # logger.info(f"UMD por función '{s}'")
            return (-2, total_cuant)

        if is_code(s):
            # logger.info(f"Code Por función: {s}")
            return (-1, total_cuant)

        encoders = text_encode(s.lower(), ["all"])
        
        dense_mean = float(sum(encoders[0]) / total_text)
        morphology_mean = float(sum(encoders[1]) / total_text)

        if dense_mean > density_thr[1]:
            # logger.info(f"DESC por codificación cuant: '{s}'")
            return (0, total_cuant)
        
        if dense_mean < density_thr[0]:
            if s.isnumeric() or any(c in VALID_NUM_PUNT_CHARS for c in s):
                # logger.info(f"NUME por codificación cuant: '{s}'")
                return (1, total_cuant)
            
                # logger.info(f"CODE por codificación cuant: '{s}'")
            return (-1, total_cuant)  # numeric
        
        if dense_mean < density_thr[1] and morphology_mean > morph_thr[0]:
            # logger.info(f"Code por codificación: '{s}'")
            return (-1, total_cuant)  # code
            
            # logger.info(f"DESC por rareza: '{s}'")
        return (0, 0)  # descriptive

    for pid, polygon in polygons.items():    
        s = polygon.ocr_text or ""
        s = s.strip()
        if not s:
            continue
        
        kf = polygon.key_field or None
        if kf or kf is not None:
            # logger.info(f"KeyField existente, no se clasifica {pid}: '{s}'")
            continue

        tokens = s.split(" ")
        total_tokens = len(tokens)
        
        if not tokens or total_tokens == 0:
            continue

        elif total_tokens == 1:
            
            if s.isalpha():
                if bool(_mesure_patterns.search(s)):
                    final_results[pid] = ([-2], 0)
                    continue
                # else:
                final_results[pid] = ([0], 0)
                continue

            total_text = len(s)
            if s.isdecimal():
                if s.startswith("0"):
                    final_results[pid] = ([-1], total_text)
                    continue
                
                final_results[pid] = ([1], total_text)
                continue

            if is_quantitative(s):
                final_results[pid] = ([2], total_text)
                # logger.info(f"CUANT POR SET: '{s}'")
                continue

            if find_umd(s):
                #logger.info(f"UMD inicial: '{s}'")
                c = sum(1 for ch in s if ch in CHAR_NUM)
                final_results[pid] = ([-2], c)
                continue
            
            if is_code(s):
                #logger.info(f"CODE INICIAL: '{s}")
                c = sum(1 for ch in s if ch in CHAR_NUM)
                final_results[pid] = ([-1], c)
                continue
            else:
                # logger.info(f"Token sin clasificación: '{s}'")
                t_class, t_cuant = classify_token(s)
                final_results[pid] = ([t_class], t_cuant)
                
        elif total_tokens > 1:
            token_classes: List[int] = []
            poly_total_cuant = 0
            for t in tokens:
                t_class, t_cuant = classify_token(t)
                token_classes.append(t_class)
                poly_total_cuant += t_cuant
            final_results[pid] = (token_classes, poly_total_cuant)

    return final_results