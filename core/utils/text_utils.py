import re
import logging
from typing import List, Tuple, Dict, Pattern, Any, Optional
from core.utils.math_utils import text_encode
from core.utils.data_utils import VALID_CUANT_CHARS, CHAR_NUM, ALONE_CHARS, VOWELS

logger = logging.getLogger(__name__)

ALONE_CHARS.update(CHAR_NUM)
CUANT_CHAR = CHAR_NUM.union(VALID_CUANT_CHARS)

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
# _spaces_pattern: Pattern[str] = re.compile(r'\s+', re.IGNORECASE)

# Siglas/Acrónimos
_acronym_pattern: Pattern[str] = re.compile(r'^(?:(?:[A-Za-z]\.){1,}[A-Za-z]\.?|sa|cv|mn)(?:[:;,.])?$', re.IGNORECASE)
# _bad_title: Pattern[str] = re.compile(r'^([A-Za-z0-9])(?: [A-Za-z0-9])+$', re.IGNORECASE)

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
    # Años
    re.compile(r'\b(199\d|20\d{2})\b', re.IGNORECASE)
]

_mass_pattern = re.compile(rf'\b(kg|gr|grs|mg|lb|lbs|oz|ton)\b', re.IGNORECASE) # Unidades de más de una letra pueden ir solas
_vol_pattern = re.compile(rf'\b(lt|ltr|ltrs|lts|ml|cc|gal)\b', re.IGNORECASE)  # Unidades de más de una letra pueden ir solas
_len_pattern = re.compile(r'\b(cm|mm|km|in|ft|mts?|m2|m\^2|m²|cm2|cm\^2|cm²|km2|km\^2|km²)\b', re.IGNORECASE) # Más de una letra pueden ir solas
_size_pattern = re.compile(r'\b(gde|med|ch|paq)\b', re.IGNORECASE)

_cuantity_str = r'\b[Cc]\s*/\b'
_semifraction_pattern = re.compile(r'(?<![A-Za-z0-9])/\d+\b', re.IGNORECASE)
_semi_c_fraction = re.compile(rf'{_cuantity_str}\d*', re.IGNORECASE) # Cantidad: C/ o C/ con número, o solo C/

_fraction_pattern = re.compile("|".join(p.pattern for p in [_semi_c_fraction, _semifraction_pattern]), re.IGNORECASE)
_mesure_patterns = re.compile("|".join(p.pattern for p in [_size_pattern, _mass_pattern, _vol_pattern, _len_pattern]), re.IGNORECASE)

_umd_patterns_list: List[Pattern[str]] = [
    re.compile(rf'\b\d+([.,]\d+)?\s*g\b', re.IGNORECASE), # Solo 'g' requiere número antes
    re.compile(rf'\b\d+([.,]\d+)?\s*l\b', re.IGNORECASE),  # Solo 'l' requiere número antes
    re.compile(rf'\b\d+([.,]\d+)?\s*m\b', re.IGNORECASE),  # Solo 'm' requiere número antes    
    _fraction_pattern,
    _mesure_patterns,
    re.compile(r'\b[1-9]\d{0,2}\s*/\s*[1-9]\d{0,2}\b'),
    re.compile(r'\b\d+(?:\s*[xX]\s*\d+)+\b') # Dimensiones (10x20)
]

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

_rfc_acronyms: Pattern[str] = re.compile(r'\b(R\.?F\.?C\.?)\b', re.IGNORECASE)
_rfc_key_pattern: Pattern[str] = re.compile(r'^([A-ZÑ]{3,4})\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])[A-Z0-9]{3}$', re.IGNORECASE)
_rfc_patterns: Pattern[str] = re.compile("|".join(p.pattern for p in [_rfc_key_pattern, _rfc_acronyms]), re.IGNORECASE)

_iva_pattern: Pattern[str] = re.compile(r'\b(I\.?V\.?A\.?)\b', re.IGNORECASE)
_date_patterns = re.compile("|".join(p.pattern for p in _date_patterns_list), re.IGNORECASE)

def validate_text(text: str) -> bool :
    """valida que un string contenga caracteres válidos y que no esté vacío"""
    total_txt = len(text)
    # Si tiene más de un carácter, debe tener al menos un alfanumérico
    if total_txt > 1:
        return any(char.isalnum() for char in text)
    # Si es un solo carácter, debe ser válido (número o en ALONE_CHARS)
    if total_txt == 1:
        return text in ALONE_CHARS
    return False

def is_code(s: str) -> bool:
    if not s or len(s) < 2:
        return False
    if not any(c.isalnum() for c in s):
        return False
    if s.isalpha():
        return False
    if bool(_numeric_code.match(s)):
        logger.debug(f"Código comienza n 0")
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
        if not s or not any(c.isalnum() for c in s):
            return False 
        return bool(_fraction_pattern.fullmatch(s)) or bool(_umd_patterns.search(s))
    except TypeError as e:
        logger.warning(f"Error buscando unidades de medida: {e}", exc_info=True)
    return False
    
def is_quantitative(text: str) -> bool:
    """
    Válida rapidamente si un string contiene caracteres válidos para ser cuantitativo.
    """
    if not text or len(text) < 3 or text.isdecimal():
        return False
        
    valid_cuant = all(c in CUANT_CHAR for c in text) if any(c.isdecimal() for c in text) else False
    return valid_cuant or bool(_quant_runs_patterns.fullmatch(text))

def contains_quantitative(text: str) -> bool:
    """
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
    
    return " ".join(result_parts)
    # return _zeros_pattern.sub("0", quants).strip()

def clean_cuant(text: str) -> str:
    """Normaliza texto para Decimal"""
    text_0 = _zeros_pattern.sub("0", text).strip()
    return _clean_currency.sub('', text_0).strip()

def punct_strip(text: str) -> str:
    """
    Elimina los caracteres de puntuación definidos en _punt_split_pattern
    que se encuentran al inicio y al final de cada token en el texto.
    """
    if not text:
        return ""
        
    if is_acronym(text):
        # logger.info(f"Acromimo: {text}")
        return text.strip()

    tokens = text.split()
    processed_tokens: List[str] = []
    # Patrón para encontrar puntuación al inicio o al final de un token
    for t in tokens:
        # Elimina la puntuación del inicio y del final del token
        cleaned_token = _edge_punt_pattern.sub("", t).strip()
        processed_tokens.append(cleaned_token)

    return " ".join(processed_tokens)

def separate_punt(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    
    if is_acronym(text):
        # logger.info(f"Acromimo: {text}")
        return text
   
    processed_tokens: List[str] = []
    tokens = text.split()
    for t in tokens:
        # Mantiene intactas horas y cuantitativos puros; limpia los tokens mixtos.
        if not bool(_hour_pattern.fullmatch(t)) and not is_quantitative(t):
            cleaned_token = _punt_split_pattern.sub(" ", t)
            processed_tokens.append(cleaned_token)
        else:
            # Si es una hora, se mantiene intacta
            processed_tokens.append(t)

    # Une los tokens y usa space_removal para normalizar todos los espacios
    return space_removal(" ".join(processed_tokens))

def space_removal(text: str) -> str:
    """
    - Normaliza espacios múltiples y limpia bordes.
    - Si no hay espacios, devuelve el texto tal cual.
    - Si hay espacios pero nunca dos seguidos, devuelve text.strip().
    - Si hay dos o más espacios seguidos, normaliza.
    """
    if not text:
        return ""
    if " " not in text:
        return text
    if "  " not in text:
        return text.strip()
    return " ".join(text.split())

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
    no_cuants = 0
    has_cuants = 0
    encoded = 0
    mixed = 0
    no_clas = 0

    def classify_token(s: str) -> Tuple[int, int]:
        nonlocal no_cuants, has_cuants, encoded, mixed, no_clas
        if not s:
            # logger.info(f"No existente: '{s}'")
            no_clas += 1
            return (-1, 0)

        if not any(c.isalnum() for c in s):
            no_clas += 1
            # logger.info(f"No alfanumérico: '{s}¿")
            return (-1, 0)

        total_text = len(s)
        total_cuant = sum(1 for ch in s if ch in CUANT_CHAR) if any(c.isdigit() for c in s) else 0
        
        if total_cuant == 0:
            if not any(c.isalpha() for c in s):
                no_clas += 1
                return (-1, total_cuant)

            elif total_text == 1:
                logger.debug(f"DESC por tamaño: '{s}'")
                no_cuants += 1
                return (1, total_cuant)

            if bool(_semi_c_fraction.fullmatch(s)) or bool(_mesure_patterns.fullmatch(s)):
                logger.debug(f"UMD por regex: '{s}'")
                no_cuants += 1
                return (2, total_cuant)

            if not any(c in VOWELS for c in s):
                if find_umd(s):
                    logger.debug(f"UMD sin vocales: '{s}'")
                    no_cuants += 1
                    return (2, total_cuant)
                    
                if total_text > 2:
                    logger.debug(f"CODE sin vocales: '{s}'")
                    no_cuants += 1
                    return (3, total_cuant)
                
            logger.debug(f"DESC por sobrante: '{s}'")
            no_cuants += 1
            return (1, total_cuant)

        if total_cuant == total_text:
            if total_text < 3:
                logger.debug(f"NUM por único: '{s}'")
                has_cuants += 1
                return (5, total_cuant)

            elif s.startswith("0"):
                logger.debug(f"CODE por inicio 0: '{s}'")
                has_cuants += 1
                return (3, total_cuant)
                
            if s.isdecimal():
                logger.debug(f"NUM por decimal: '{s}'")
                has_cuants += 1
                return (5, total_cuant)
                
            if contains_quantitative(s):
                logger.debug(f"CUANT por validación: '{s}'")
                has_cuants += 1
                return (4, total_cuant)
            logger.debug(f"NUM por descarte en conteo: '{s}'")
            has_cuants += 1
            return (5, total_cuant)

        if contains_quantitative(s):
            logger.debug(f"CUANT mixto: '{s}'")
            mixed += 1
            return (4, total_cuant)

        if find_umd(s):
            logger.debug(f"UMD mixto: '{s}'")
            mixed += 1
            return (2, total_cuant)

        if is_code(s):
            logger.debug(f"CODE mixto: '{s}'")
            mixed += 1
            return (3, total_cuant)
        
        encoders = text_encode(s.lower(), ["all"])
        dense_mean = float(sum(encoders[0]) / total_text)
        morphology_mean = float(sum(encoders[1]) / total_text)

        if dense_mean > density_thr[1]:
            logger.debug(f"DESC por codificacion: '{s}'")
            encoded += 1
            return (1, total_cuant)

        if dense_mean < density_thr[0]:
            if (total_cuant / total_text) > 0.687:
                logger.debug(f"NUM por codificacion: '{s}'")
                encoded += 1
                return (5, total_cuant)
            logger.debug(f"CODE por descarte de codificacion NUM: '{s}'")
            encoded += 1
            return (3, total_cuant)

        if dense_mean < density_thr[1] and morphology_mean > morph_thr[0]:
            logger.debug(f"CODE por codificacion: '{s}'")
            encoded += 1
            return (3, total_cuant)
        
        logger.debug(f"Poligono sin clasificación, será descriptiva: '{s}'")
        encoded += 1
        no_clas += 1
        return (1, total_cuant)

    for pid, polygon in polygons.items():
        kf = polygon.key_field or None
        if kf or kf is not None:
            final_results[pid] = ([0], 0)
            logger.debug(f"KeyField existente, no se clasifica '{polygon.ocr_text or ""}'")
            no_clas += 1
            continue
        
        s = polygon.ocr_text or ""
        s = s.strip()
        if not s:
            final_results[pid] = ([-1], 0)
            no_clas += 1
            # logger.info(f"No existente: {s}")
            continue

        tokens = s.split(" ")
        total_tokens = len(tokens) # Cantidad de palabras
        
        if not tokens or 0 >= total_tokens:
            final_results[pid] = ([-1], 0)
            no_clas += 1
            # logger.info(f"No valido: {s}")
            continue            
                
        token_classes: List[int] = []
        poly_total_cuant = 0
        for t in tokens:
            t_class, t_cuant = classify_token(t)
            token_classes.append(t_class)
            poly_total_cuant += t_cuant
        final_results[pid] = (token_classes, poly_total_cuant)
    
    # logger.info(f"TOTAL CLASIFICADOS SIN CUANTITATIVOS: '{no_cuants}', SIN CUANTS: {has_cuants}, CODIFICADOS: {encoded}, MIXTOS: {mixed}")
    return final_results