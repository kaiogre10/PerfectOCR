import re
import logging
from typing import List, Tuple, Dict, Pattern, Any, Set
from core.utils.math_utils import text_encode
from core.utils.data_utils import CHAR_NUM, ALONE_CHARS, VALID_NUM_PUNT_CHARS

logger = logging.getLogger(__name__)

_zeros_str = r'[O0QD]'
_base_date_num_str = r'[0123O][0-9O]'
# _base_date_num: Pattern[str] = re.compile(rf'^{_base_date_num_str}$')
_zeros_pattern = re.compile(_zeros_str, re.IGNORECASE)

# Patrón para secuencias especiales de 2 o más caracteres no alfanuméricos (excluyendo espacio, $, /,)
_secuence_pattern: Pattern[str] = re.compile(r'[^a-zA-Z0-9/\s/$]{2,}', re.IGNORECASE)
_sequence_middle_pattern: Pattern[str] = re.compile(r'(?<=[a-zA-Z0-9/$])[^a-zA-Z0-9/\s/$]{2,}(?=[a-zA-Z0-9/$])', re.IGNORECASE)

# _numeric_separator: Pattern[str] =  re.compile(r'^([$\u00A2]?\s*)(-?\d[\d.,]*)(\s*[$\u00A2]?)$', re.IGNORECASE)
_hour_pattern: Pattern[str] = re.compile(rf'\b{_base_date_num_str}:[0-5O][0-9O](?::[0-5O][0-9O])?\b', re.IGNORECASE)
_punt_split_pattern: Pattern[str] = re.compile(r"[*_'`=.,:;&-]")
_edge_punt_pattern = re.compile(rf'^({_punt_split_pattern.pattern}+)|({_punt_split_pattern.pattern}+)$', re.IGNORECASE)

# Espacios múltiples
_spaces_pattern: Pattern[str] = re.compile(r'\s+')

# Siglas/Acrónimos
_acronym_pattern: Pattern[str] = re.compile(r'^(?:(?:[A-Za-z]\.){1,}[A-Za-z]\.?|sa|cv|no)(?:[:;,.])?$', re.IGNORECASE)

# Datos Globales
_numeric_code: Pattern[str] = re.compile(rf'^{_zeros_str}\d+$', re.IGNORECASE)
_phone_number: Pattern[str] = re.compile(r'^\d{10}$')
_mail_pattern: Pattern[str] = re.compile(r'.*@.+', re.IGNORECASE)
_cp_pattern: Pattern[str] = re.compile(r'^(?:C\.?P\.?\s*)\d{5}$', re.IGNORECASE)

_code_patterns: Pattern[str] = re.compile("|".join(p.pattern for p in [_phone_number, _mail_pattern, _cp_pattern]), re.IGNORECASE)

# Fecha
_date_patterns_list: List[Pattern[str]] = [
    # Día + mes en letras + año en un solo string OCR (ej. "21 mar 2023")
    re.compile(
        rf'\b{_base_date_num_str}\s+(?:ene(?:ro)?|feb(?:rero)?|mar(?:zo)?|abr(?:il)?|may(?:o)?|jun(?:io)?|jul(?:io)?|ago(?:s(?:to)?)?|sep(?:t(?:iembre)?)?|oct(?:ubre)?|nov(?:iembre)?|dic(?:iembre)?)\s+(?:19\d{{2}}|20\d{{2}})\b',
        re.IGNORECASE,
    ),
    # Fechas completas y día/mes: Usa el bloque base para evitar confundirse con fracciones/cuantitativos
    re.compile(rf'\b{_base_date_num_str}[\s\/\-]{_base_date_num_str}(?:[\s\/\-](?:\d{{2,4}}))?\b', re.IGNORECASE),
    # Meses (palabras)
    re.compile(r'\b(ene(ro)?|feb(rero)?|mar(zo)?|abr(il)?|may(o)?|jun(io)?|jul(io)?|ago(s(to)?)?|sep(t(iembre)?)?|oct(ubre)?|nov(iembre)?|dic(iembre)?)\b', re.IGNORECASE),
    # Años
    re.compile(r'\b(199\d|20\d{2})\b', re.IGNORECASE)
]

_mass_pattern = r'(kg?|g(r)?(s)?|mg|lb(s)?|oz|ton)\b'
_vol_pattern = r'(l(t(r)?(s)?)?|ltrs?|lts?|ml|cc|gal)\b'
_len_pattern = r'(m(t(s)?)?|cm|mm|km|in|ft)\b'

_umd_patterns_list: List[Pattern[str]] = [
    # Masas: kg, g, mg, lb, oz, ton. Incluye variaciones de OCR como kgr.
    re.compile(rf'\b\d*([.,]\d+)?\s*{_mass_pattern}', re.IGNORECASE),
    # Volúmenes: l, ml, cc, gal. Incluye variaciones como lt, ltr.
    re.compile(rf'\b\d*([.,]\d+)?\s*{_vol_pattern}', re.IGNORECASE),
    # Cantidad: C/ o C/ con número.
    re.compile(r'\b[Cc]\s*/\s*\d*\b', re.IGNORECASE),
    # Longitudes y Áreas: m, cm, mm, km, in, ft, pulg. Detecta la unidad sola o con número. Soporta m2, m^2, m² para áreas.
    re.compile(r'\b(\d+([.,]\d+)?\s*)?(m(t(s)?)?|cm|mm|km|in|ft|m(t)?(\^2|2|²)|cm(\^2|2|²)|km(\^2|2|²))\b', re.IGNORECASE),

    re.compile(r'\b[1-9]\d{0,2}\s*/\s*[1-9]\d{0,2}\b'),
    # Fracciones (1/2 kg, 1/4).
  #  re.compile(r'\b\d+\s*/\s*\d+\b'),
    # Dimensiones (10x20)
    re.compile(r'\b\d+(?:\s*[xX]\s*\d+)+\b')
]

_size_str = r'\b(gde|med|ch|paq)\b'
#size_pattern: Pattern[str] = re.compile(_size_str, re.IGNORECASE)
_mesure_patterns= re.compile("|".join(p for p in [_size_str, _mass_pattern, _vol_pattern, _len_pattern]), re.IGNORECASE)

_umd_patterns = re.compile("|".join(p.pattern for p in _umd_patterns_list), re.IGNORECASE)

# Define los patrones como strings
digit_pattern = r"[0-9oOQ]"
currency_pattern = r"[$]"
_clean_currency_pattern = (rf'\b[{currency_pattern},]')
# Patrón: S al inicio, al menos 3 dígitos entre la S y un punto o coma
# _s_correct_pattern = re.compile(r'^S\d{3,}[.,]', re.IGNORECASE)

# _currency_stick_pattern: Pattern[str] = re.compile(r'([a-zA-Z])([\$])')

# Compila los patrones base
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
_token = re.compile(_token_pattern)

# Patrón: Montos con símbolo al inicio ($ 80.50)
_start_pattern = rf"^{currency_pattern}\s*{_amount_body_pattern}$"
_start = re.compile(_start_pattern)

# Patrón: Monto con símbolo en medio (80 $ 50)
_middle_pattern = rf"^{_amount_body_pattern}\s*{currency_pattern}\s*{_amount_body_pattern}$"
_middle = re.compile(_middle_pattern)

# Patrón: Múltiples montos seguidos de símbolo ($100 $200)
_multi_pattern = rf"^(?:\s*{currency_pattern}\s*{_amount_body_pattern}\s*){{2,}}$"
_multi = re.compile(_multi_pattern)

# Patrón: Decimales grandes tipo 1,230.50 (sin $)
_decimal = re.compile(r"^\d{1,3}(?:[.,]\d{3})*[.,]\d{2,}$")

_quant_runs_patterns = re.compile("|".join(p.pattern for p in [_decimal, _start, _middle, _multi]), re.IGNORECASE)

# Patrón: Monto terminado en símbolo (80.00 $)
# _end_pattern = rf"^{_amount_body_pattern}\s*{currency_pattern}$"
# _end = re.compile(_end_pattern, re.IGNORECASE)

# Extrae solo los dígitos (sin formato decimal)
# _digits = re.compile(r"\d+")

# Detecta terminaciones típicas de dinero (.00 ó ,00)
# _end_quants = re.compile(r'[.,]00$', re.IGNORECASE)

# Patrón equivalente a _split, pero requiere $ al inicio y una cantidad
_split_pattern = rf"{currency_pattern}\s*{_amount_body_pattern}"
_split = re.compile(_split_pattern, re.IGNORECASE)

_rfc_acronyms: Pattern[str] = re.compile(r'\b(R\.?F\.?C\.?)\b')
_rfc_pattern: Pattern[str] = re.compile(r'^([A-ZÑ]{3,4})\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])[A-Z0-9]{3}$', re.IGNORECASE)


IVA_PATTERN: Pattern[str] = re.compile(r'\b(I\.?V\.?A\.?)\b', re.IGNORECASE)
DATE_PATTENRS = re.compile("|".join(p.pattern for p in _date_patterns_list), re.IGNORECASE)

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
    elif total_txt == 1:
        return text.isdecimal() or text in ALONE_CHARS
    else:
        return False

def is_code(s: str) -> bool:
    if not any(c.isalpha() for c in s):
        return False
    elif s.isalpha():
        return False
    elif bool(_numeric_code.match(s)):
        logger.info(f"Código comienza n 0")
        return True
    else:
        return bool(_code_patterns.search(s))

def find_key_data(s: str, activate_func: List[bool]) -> int:
    """
    Busca fecha (9), RFC (7) o IVA (8) en el texto crudo del polígono.
    activate_func: [fecha_ya_encontrada, rfc_ya_encontrado, iva_ya_encontrado];
    se pone True en el índice correspondiente al devolver un key_field distinto de 0.
    Prioridad: fecha > RFC > IVA (solo un tipo por llamada).
    """
    try:
        s = s.strip()
        if not any(c.isalnum() for c in s):
            return 0

        if not activate_func[0] and bool(DATE_PATTENRS.search(s)):
            activate_func[0] = True
            return 9

        if not activate_func[1] and bool(RFC_PATTERNS.search(s)):
            activate_func[1] = True
            return 7

        if not activate_func[2] and bool(IVA_PATTERN.search(s)):
            activate_func[2] = True
            return 8

        return 0

    except ValueError as e:
        logger.warning(f"Error buscando datos globales: {e}", exc_info=True)
    return 0

def find_date(s: str) -> bool:
    try:
        if s.isalpha():
            return False
        else:
            return bool(DATE_PATTENRS.search(s))

    except TypeError as e:
        logger.warning(f"Error buscando fecha: {e}", exc_info=True)
    return False

def find_rfc(s: str) -> bool:
    try:
        if len(s) < 12:
            return False
        
        elif not any(c.isdecimal() for c in s):
            return False
        
        else:
            return bool(RFC_PATTERNS.search(s))

    except TypeError as e:
        logger.warning(f"Error buscando RFC: {e}", exc_info=True)
    return False

def find_iva(s: str) -> bool:
    try:
        if not any(c.isalnum() for c in s):
            return False
        
        return bool(IVA_PATTERN.search(s))
    except TypeError as e:
        logger.warning(f"Error Buscando IVA: {e}", exc_info=True)
    return False

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
    
    return all(c in char_num_point for c in text) or bool(_quant_runs_patterns.fullmatch(text))

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
    Aísla cuantitativos SÓLO si están pegados a otros caracteres (ruido o texto).
    Si ya están separados por espacios, no modifica el texto.
    """
    if not text:
        return ""
    
    elif text.isalpha():
        return text
    
    elif not contains_quantitative(text):
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

        result_parts.append(result)
    
    quants = " ".join(result_parts)
    return _zeros_pattern.sub("0", quants).strip()

def clean_cuant(text: str) -> str:
    """Normaliza texto para Decimal"""
    text_0 = _zeros_pattern.sub("0", text).strip()
    return _clean_currency.sub('', text_0).strip()

def clean_punct(text: str) -> str:
    """
    Elimina los caracteres de puntuación definidos en _punt_split_pattern
    que se encuentran al inicio y al final de cada token en el texto.
    """
    if not text:
        return ""

    tokens = text.split()
    processed_tokens: List[str] = []
    # Patrón para encontrar puntuación al inicio o al final de un token
    for t in tokens:
        # Elimina la puntuación del inicio y del final del token
        cleaned_token = _edge_punt_pattern.sub('', t)
        processed_tokens.append(cleaned_token)

    return " ".join(processed_tokens).strip()

def separate_punt(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    
    elif text.isalnum():
        return text

    elif is_acronym(text):
        return text
   
    tokens = text.split()
    processed_tokens: List[str] = []
    for t in tokens:
        t = t.strip()
        # Mantiene intactas horas y cuantitativos puros; limpia los tokens mixtos.
        if not _hour_pattern.fullmatch(t) and not is_quantitative(t):
            cleaned_token = _punt_split_pattern.sub(' ', t).strip()
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
    
    elif " " not in text or text == text.strip():
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
            if bool(_mesure_patterns.match(s)):
                return (-2, 0)
            else:
                return (0, 0)
    
        elif s.isdecimal():
            if total_text > 1 and s.startswith("0"):
                return (-1, 0)
            else:
                return (1, total_text)

        total_cuant = sum(1 for ch in s if ch in CHAR_NUM)

        if total_cuant == 0:
            if bool(_mesure_patterns.search(s)):
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

        tokens = s.split()
        total_tokens = len(tokens)
        if not tokens or total_tokens == 0:
            continue

        elif total_tokens >= 2:
            token_classes: List[int] = []
            poly_total_cuant = 0
            for t in tokens:
                t_class, t_cuant = classify_token(t.strip())
                token_classes.append(t_class)
                poly_total_cuant += t_cuant
            final_results[pid] = (token_classes, poly_total_cuant)
        
        else:
            if s.isalpha():
                if bool(_mesure_patterns.match(s)):
                    final_results[pid] = ([-2], 0)
                else:
                    final_results[pid] = ([0], 0)

            total_text = len(s)
            if s.isdecimal():
                if total_text > 1 and s.startswith("0"):
                    final_results[pid] = ([-1], total_text)
                else:
                    final_results[pid] = ([1], total_text)

            elif is_quantitative(s):
                final_results[pid] = ([2], total_text)
                # logger.info(f"CUANT POR SET: '{s}'")

            elif find_umd(s):
                #logger.info(f"UMD inicial: '{s}'")
                c = sum(1 for ch in s if ch in CHAR_NUM)
                final_results[pid] = ([-2], c)
            
            elif is_code(s):
                #logger.info(f"CODE INICIAL: '{s}")
                c = sum(1 for ch in s if ch in CHAR_NUM)
                final_results[pid] = ([-1], c)

            else:
                # logger.info(f"Token sin clasificación: '{s}'")
                t_class, t_cuant = classify_token(s)
                final_results[pid] = ([t_class], t_cuant)

    return final_results