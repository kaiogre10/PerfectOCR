# PerfectOCR/core/utils/text_utils.py
import logging
import unicodedata
from typing import List, Tuple, Dict, Any, Optional
from core.utils.math_utils import text_encode, round_2_decimal_vals
from core.utils.data_utils import VALID_ALONE_CHARS, VOWELS, VALID_CUANT_CHARS, CHAR_NUM
from core.utils.patterns import rfc_key_pattern, numeric_code, acronym_pattern, acromin_currency_pattern, cion_search_patt, suffix_pattern, cion_str, con_suffix_pattern, con_search_patt, con_str, umd_patterns, date_patterns, umd_cor, amount_fract, zeros_pattern, fraction_pattern, rfc_patterns, iva_patterns, phone_number, mail_pattern, cp_pattern, quant_runs_patterns, token, valid_cuant_pattern, split, monetary_pattern, clean_currency, edge_punt_pattern, hour_pattern, punt_split_pattern, sequence_middle_pattern, secuence_pattern, labels_pattern, size_pattern, semi_c_fraction, id_prov_pattern, los_str, los_search_patt, los_suffix_pattern, measure_fractions, c_dash_fraction_pattern, end_cuants,swap_term_cuant

logger = logging.getLogger(__name__)

density_thr = (23.7, 103.7)
morph_thr = (-0.297, 0.337)

_swap_term_cuant = swap_term_cuant
_end_cuants = end_cuants
_c_dash_fraction_pattern = c_dash_fraction_pattern
_measure_fractions = measure_fractions
_rfc_key_pattern = rfc_key_pattern
_numeric_code = numeric_code
_acronym_pattern = acronym_pattern
_acromin_currency_pattern = acromin_currency_pattern
_cion_search_patt = cion_search_patt
_suffix_pattern = suffix_pattern
_cion_str = cion_str
_con_suffix_pattern = con_suffix_pattern
_con_search_patt = con_search_patt
_con_str = con_str
_los_str = los_str
_los_search_patt = los_search_patt
_los_suffix_pattern = los_suffix_pattern
_umd_patterns = umd_patterns
_date_patterns = date_patterns
_umd_cor = umd_cor
_amount_fract = amount_fract
_zeros_pattern = zeros_pattern
_fraction_pattern = fraction_pattern
_rfc_patterns = rfc_patterns
_iva_patterns = iva_patterns
_phone_number = phone_number
_mail_pattern = mail_pattern
_cp_pattern = cp_pattern
_quant_runs_patterns = quant_runs_patterns
_token = token
_valid_cuant_pattern = valid_cuant_pattern
_split = split
_monetary_pattern = monetary_pattern
_clean_currency = clean_currency
_edge_punt_pattern = edge_punt_pattern
_hour_pattern = hour_pattern
_punt_split_pattern = punt_split_pattern
_sequence_middle_pattern = sequence_middle_pattern
_secuence_pattern = secuence_pattern
_labels_pattern = labels_pattern
_size_pattern = size_pattern
_semi_c_fraction = semi_c_fraction
_id_prov_pattern = id_prov_pattern

alone_chars = VALID_ALONE_CHARS
char_num = CHAR_NUM
cuant_chars = frozenset(char_num.union(VALID_CUANT_CHARS))
vowels = VOWELS

def normalice_text(s: str, hard_norm: Optional[bool] = False) -> str:
    """"Normaliza texto eliminando apóstrofes, tildes, diéresis. NO ELIMINA CARACTERES DE NINGÚN TIPO, MISMO LEN() EN INPUT Y OUTPUT"""
    if not s:
        return ""
    norm_text = "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")
    if norm_text and hard_norm:
        hard_text =  unicodedata.normalize('NFKD', norm_text).encode('ascii', 'ignore').decode('utf-8')
        return "" if not hard_text else hard_text
    return norm_text if norm_text else ""
    
def get_rfc(s: str) -> str:
    if not s:
        return ""
    match = _rfc_key_pattern.search(s.strip())
    return match.group(0) if match else ""

def is_code(s: str) -> bool:
    if not s or len(s) < 3:
        return False
    if not any(c.isalnum() for c in s):
        return False
    if s.isalpha():
        return False
    return bool(_numeric_code.search(s))

def is_acronym(text: str) -> bool:
    if not text or text.isnumeric() or text.isalpha():
        return False
    elif bool(_acronym_pattern.search(text)) or bool(_acromin_currency_pattern.search(text)):
        return True
    else:
        return False

def correct_subfix(text: str) -> str:
    if not text:
        return ""

    elif len(text) < 3 or text.isnumeric():
        return text
    
    elif not bool(_cion_search_patt.search(text)) and bool(_suffix_pattern.search(text)):
        return _suffix_pattern.sub(_cion_str, text)

    elif not bool(_con_search_patt.search(text)) and bool(_con_suffix_pattern.search(text)):
        return _con_suffix_pattern.sub(_con_str, text)

    elif not bool(_los_search_patt.search(text)) and bool(_los_suffix_pattern.search(text)):
        return _los_suffix_pattern.sub(_los_str, text)
    else:
        return text

def contains_umd(s: str) -> bool:
    """Valida si un string contiene UMD"""
    return bool(_umd_patterns.search(s))

def find_umd(s: str) -> str:
    """En un string completo inserta espacios en los bordes de cada UMD para separar subcadenas con split solo si hay UMD."""
    if not s:
        return ""
    
    elif bool(_date_patterns.search(s)):
        return s

    if not s.endswith("0") and not s[-1].isdecimal() and _umd_cor.search(s):
        new_s = s[:-1] + "0"
        if not new_s.isdecimal() and not is_quantitative(new_s):
            return new_s
        s = new_s

    if bool(_measure_fractions.fullmatch(s)) or bool(_amount_fract.fullmatch(s)):
        if "7" in s and not bool(_c_dash_fraction_pattern.search(s)):
            s = s.replace("7", "/", 1)
            s = _zeros_pattern.sub("0", s)
    
    if _fraction_pattern.fullmatch(s):
        return s

    intervals: List[Tuple[int, int]] = [(m.start(), m.end()) for m in _umd_patterns.finditer(s)]
    if not intervals:
        return s

    # Solo fusionar solapamientos reales (start < fin_anterior). Si start == fin_anterior son dos UMD pegados; no fusionar para poder insertar espacio entre ellos.
    merged: List[Tuple[int, int]] = []
    for start, end in sorted(intervals, key=lambda t: t[0]):
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    parts: List[str] = []
    pos = 0
    for start, end in merged:
        gap = s[pos:start]
        if gap:
            parts.append(gap)
        if start > 0:
            if gap:
                need_space = not gap[-1].isspace()
            else:
                need_space = bool(parts) and not parts[-1][-1].isspace()
            if need_space:
                parts.append(" ")
        parts.append(s[start:end])
        pos = end
    tail = s[pos:]
    if tail:
        if parts and not parts[-1][-1].isspace() and not tail[0].isspace():
            parts.append(" ")
        parts.append(tail)
    return "".join(parts)

def find_key_data(s: str, activate_func: List[bool]) -> Optional[int]:
    if not any(c.isalnum() for c in s):
        return None

    if not activate_func[0] and bool(_date_patterns.search(s)):
        activate_func[0] = True
        return 9

    if not activate_func[1] and bool(_rfc_patterns.search(s)):
        activate_func[1] = True
        return 7

    if not activate_func[2] and bool(_iva_patterns.search(s)):
        activate_func[2] = True
        return 8

    if not activate_func[3] and bool(_phone_number.search(s)):
        activate_func[3] = True
        return 10

    if not activate_func[4] and bool(_mail_pattern.search(s)):
        activate_func[4] = True
        return 11

    if not activate_func[5] and bool(_acromin_currency_pattern.search(s)):
        activate_func[5] = True
        return 0
    
    if not activate_func[6] and bool(_cp_pattern.search(s)):
        activate_func[6] = True
        return 12
    return None

def validate_quant_chars(text: str) -> bool:
    """Valida todos si todos los caracteres de un string pertenecen a 'cuant_chars' cuantitativos"""
    valid = False
    for c in text:
        if c not in cuant_chars:
            return False
        if c in char_num:
            valid = True
    return valid

def validate_quant_pattern(text: str) -> bool:
    """Verifica si hay match completo de un patrón"""
    return bool(_quant_runs_patterns.fullmatch(text)) 
    
def is_quantitative(text: str) -> bool:
    """Válida rapidamente si un string es cuantitativo"""
    if len(text) < 3 or text.isdecimal():
        return False
    return bool(validate_quant_pattern(text) or validate_quant_chars(text))

def contains_quantitative(text: str) -> bool:
    """Devuelve True si encuentra algún sub-string cuantitativo en el texto."""
    if not text:
        return False
    match = _token.search(text)
    return bool(match and is_quantitative(match.group(0)))

def get_cuants(text: str) -> str:
    """Aísla cuantitativos SÓLO si están pegados a otros caracteres (ruido o texto). Si ya están separados por espacios, no modifica el texto."""
    if len(text) < 3 or text.isdecimal() or text.isalpha():
        return text

    words = text.split(" ")
    result_parts: List[str] = []
    for word in words:
        monetary_count = word.count("$")
        
        if bool(_valid_cuant_pattern.search(word)):
            logger.info(f"valid_cuant_pattern: '{word}'")
            word_c = format_cuant(word)
            word = "$" + word_c
            logger.info(f"valid_cuant_pattern 2: '{word}'")

        if monetary_count >= 2 and contains_quantitative(word): 
            compact = word.replace(" ", "")                                          # $10.50.$31.50 | $10.50$31.50
            chunks = [m.group(0).replace(" ", "") for m in _split.finditer(compact)] # $10.50.$31.50 | $10.50 $31.50
            
            total_chunks = len(chunks)
            compact_chunks = "".join(chunks)
            
            if total_chunks == 1:
                if compact_chunks == compact:
                    chunks = compact.replace("$", " ")
                    result_parts.append(" ".join(chunks.split()))
                    continue
                
                result_parts.append(" ".join(chunks))
                continue
            
            if total_chunks >= 2:                            # $10.50.$31.50 == 1| $10.50 $31.50 == 2
                if compact_chunks != compact:                # Siempre True
                    result_parts.append(" ".join(chunks))
                    continue
            
        elif monetary_count == 1:
            if is_quantitative(word):
                result_parts.append(word)
                continue

            elif validate_quant_pattern(word):
                word = _zeros_pattern.sub("0", word)
                result_parts.append(word)
                continue

            elif validate_quant_chars(word):
                if bool(_end_cuants.search(word)):
                    word = format_cuant(word)
                    word = _zeros_pattern.sub("0", word)
                    new_dec = round_2_decimal_vals(word)
                    word = ("$" + new_dec)
                result_parts.append(word)
                continue
            else:
                if bool(_swap_term_cuant.search(word)):
                    word = _swap_term_cuant.sub("s", word)
                    result_parts.append(word)

        elif bool(_monetary_pattern.search(word)):
            logger.info(f"MONETARY PATTERN: {word}")
            word = "$" + word[1:]

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
                needs_left_space = start > 0 and (result[start - 1].isalpha() or not result[start - 1].isdecimal())
                needs_right_space = end < len(result) and (result[end].isalpha() or not result[end].isdecimal())

                if needs_left_space or needs_right_space:
                    left_part = result[:start]
                    right_part = result[end:]
                    mid = f"{' ' if needs_left_space else ''}{tok}{' ' if needs_right_space else ''}"
                    result = left_part + mid + right_part

        result_parts.append(result)

    clean_quants =  " ".join(result_parts).strip()
    cuant_cheks = clean_quants.split(" ")
    cheks = len(cuant_cheks)
    if cheks == 1:
        return clean_quants
    elif cheks > 1:
        potencial_noise = cuant_cheks[-1]
        if len(potencial_noise) < 3 and not is_quantitative(potencial_noise):
            cuant_cheks.pop(-1)
            clean_quants =  " ".join(cuant_cheks).strip()
        return clean_quants.strip()
    else:
        return ""

def format_cuant(text: str) -> str:
    """Convierte strings numericos a cuantitativos y limpia los que ya son cuantitativos para usar Decimal"""
    if text.isdecimal():
        return (text + ".00").strip()
    text_0 = _zeros_pattern.sub("0", text)
    cuant_txt = _clean_currency.sub("", text_0).strip()
    # cuant_txt = cuant_txt.replace("$", "5").strip()
    temp_txt = cuant_txt.replace(".", "")
    if not temp_txt.isdecimal() or len(cuant_txt) > len(text):
        logger.warning(f"CARACTER INTRUSO EN '{text}' NO SE PUDO FORMATEAR")
        return text
    else:
        return cuant_txt
            
def punct_strip(text: str) -> str:
    """Elimina los caracteres de puntuación que se encuentran al inicio y al final de cada token en el texto."""
    if not text:
        return ""
        
    if validate_quant_chars(text) or is_acronym(text):
        # logger.info(f"NO SE SEPARA: {text}")
        return text.strip()
    
    return _edge_punt_pattern.sub("", text).strip()

def separate_punt(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    
    if is_acronym(text):
        return text

    tokens = text.split()
    processed_tokens: List[str] = []
    for t in tokens:
        # Mantiene intactas horas y cuantitativos puros; limpia los tokens mixtos.
        if not bool(_hour_pattern.search(t)) and not is_quantitative(t):
            cleaned_token = _punt_split_pattern.sub(" ", t)
            # logger.info(f"cleaned_token: '{t}' -> '{cleaned_token}'")
            processed_tokens.append(cleaned_token)
        else:
            # Si es una hora, se mantiene intacta
            processed_tokens.append(t)
    # Une los tokens y usa space_removal para normalizar todos los espacios
    return space_removal(" ".join(processed_tokens))

def space_removal(text: str) -> str:
    """Normaliza espacios múltiples y limpia bordes."""
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
    Conserva los caracteres sueltos válidos, pero reemplaza por un espacio las secuencias internas de símbolos, y luego limpia espacios sobrantes.
    remove_special_sequences("abc@@def!!ghi") -> 'abc def ghi'
    """
    cleaned = _sequence_middle_pattern.sub(" ", text).strip()
    cleaned = _secuence_pattern.sub("", cleaned)
    return space_removal(cleaned) if cleaned else text
    
def get_brands(text: str) -> bool:
    """Experimental: LOCALIZA SI UN STRING ES UNA MARCA COMERCIAL"""
    if len(text) != 3:
        return False
    return bool(_labels_pattern.fullmatch(text))

def clasify_words(polygons: Dict[str, Any]) -> Dict[str, Tuple[List[int], int]]:
    final_results: Dict[str, Tuple[List[int], int]] = {}
    for pid, polygon in polygons.items():
        
        if polygon.key_field is not None and 0 in polygon.semantic_clasification:
            # logger.info(f"Poligono {pid} con {polygon.key_field} keyfield existente, no se clasifica '{polygon.ocr_text or ""}'")
            final_results[pid] = ([0], 0)
            continue
        
        s = polygon.ocr_text or ""
        s = s.strip()
        if not s:
            final_results[pid] = ([-1], 0)
            # #  logger.info(f"VACIO 1ro: '{s}'")
            continue

        tokens = s.split(" ")
        total_tokens = len(tokens) # Cantidad de tokens

        if total_tokens == 1:
            t_class, t_cuant = classify_token(tokens[0])
            final_results[pid] = ([t_class], t_cuant)
    
        else:
            token_classes: List[int] = []
            poly_total_cuant = 0
            for t in tokens:

                t_class, t_cuant = classify_token(t)
                token_classes.append(t_class)
                poly_total_cuant += t_cuant
                
            final_results[pid] = (token_classes, poly_total_cuant)
            continue

    return final_results

def classify_token(s: str) -> Tuple[int, int]:
    total_text = len(s)
    total_cuant = sum(1 for ch in s if ch in cuant_chars) if any(c.isdecimal() for c in s) else 0
    if not s:
        #  logger.info(f"VACIO 3RO: {s}")
        return (-1, 0)
        
    elif not any(c.isalnum() for c in s):
        #  logger.info(f"INVÁLIDO 1: '{s}'")
        return (-1, 0)
        
    elif s.isalpha():
        if _size_pattern.search(s):
            return (2, 0)
        #  logger.info(f"ALPHA 1ro: '{s}'")
        else:
            return (1, 0)
    
    elif s.isdecimal():
        if "0" in s and bool(_numeric_code.fullmatch(s)):
            return (3, total_cuant)
        else:
            return (5, total_text)
    
    if total_cuant == 0:
        if not any(c.isalpha() for c in s):
            return (-1, 0)
        
        elif bool(_semi_c_fraction.search(s)) or bool(_amount_fract.search(s)):
            # logger.info(f"UMD por regex: '{s}'")
            return (2, 0)

        elif is_acronym(s):
            if bool(_acromin_currency_pattern.search(s)):
                # logger.info(f"ACRONIMO DEC: '{s}'")
                return (0, 0)
            else:
                return (1, 0)

        if not any(c in vowels for c in s):
            # logger.info(f"NO DETECCIÓN: {s}")
            if _fraction_pattern.search(s):
                # logger.info(f"UMD sin vocales: '{s}'")
                return (2, 0)
                
            elif total_text > 2:
                # logger.info(f"CODE sin vocales: '{s}'")
                return (3, 0)
            
        # logger.info(f"DESC por sobrante: '{s}'")
        return (1, 0)

    elif total_cuant == total_text:
        if total_text < 3:
            if s == "0":
                # #  logger.info(f"NUMERIC 0: '{s}'")
                return (5, total_cuant)
            else:
                #  logger.info(f"CODE CONSOLACIÓN: '{s}'")
                return (3, total_cuant)

        elif s.startswith("0"):
            if s.isalnum():
                #  logger.info(f"CODE por inicio 0: '{s}'")
                return (3, total_cuant)
                
            elif validate_quant_pattern(s):
                # #  logger.info(f"CUANT por inicio 0: '{s}'")
                return (4, total_cuant)
            else:
                #  logger.info(f"UMD por inicio 0: '{s}'")
                return (2, total_cuant)
            
        if is_quantitative(s):
            #  logger.info(f"CUANT por validacion: '{s}'")
            return (4, total_cuant)

        #  logger.info(f"NUM por descarte en conteo: '{s}'")
        return (5, total_cuant)
    
    if contains_quantitative(s):
        #  logger.info(f"CUANT CONTENIDO: '{s}'")
        return (4, total_cuant)

    elif bool(_acromin_currency_pattern.search(s)):
        return (0, 0)

    elif contains_umd(s):
        # logger.info(f"UMD mixto: '{s}'")
        return (2, total_cuant)

    elif is_code(s):
        # logger.info(f"CODE mixto: '{s}'")
        return (3, total_cuant)
    
    #  logger.info(fr"REBELDES: '{s}'")
    dense_mean, morphology_mean = text_encode(s.lower())
    if dense_mean < density_thr[1] and morphology_mean > morph_thr[0]:
        # logger.info(f"CODE por codificacion: '{s}'")
        return (3, total_cuant)
    
    elif dense_mean > density_thr[1]:
        logger.info(f"DESC por codificacion: '{s}'")
        return (1, total_cuant)

    if dense_mean < density_thr[0]:
        if bool(_fraction_pattern.search(s)):
            # logger.info(f"UMD por codificacion: '{s}'")
            return (2, total_cuant)

        if not any(c in ("/", ":") for c in s) and (total_cuant / total_text) > 0.687:
            #  logger.info(f"NUM por codificacion: '{s}'")
            return (5, total_cuant)
        # logger.info(f"CODE por descarte de codificacion NUM: '{s}'")
        return (3, total_cuant)
        
    elif bool(_labels_pattern.fullmatch(s)):
        # logger.info(f"DESCR MARCA: '{s}")
        return (1, total_cuant)
    
    if not any(c in vowels for c in s) and total_text > 2:
        # logger.info(f"CODE por FALLBACK: '{s}'")
        return (3, total_cuant)
        
    # logger.info(f"Poligono sin clasificación, será descriptiva: '{s}'")
    return (1, total_cuant)

def get_ids(img_name: str) -> str:
    match = _id_prov_pattern.search(img_name.strip())
    return match.group(0) if match else ""

def format_elapsed_time(seconds: float) -> str:
    """Convierte segundos a formato HH:MM:SS.ms"""
    if seconds < 60.0:
        return f"{seconds:.8f}'s"
    minutes = int((seconds % 3600) // 60)
    if minutes < 60:
        return f"{minutes:02d}:M {seconds % 60:06.3f}'s"
    else:
        return f"{int(seconds // 3600):02d}:H {minutes:02d}:M {seconds % 60:06.3f}'s"

def fast_classfier(text: str) -> Tuple[List[int], int]:
    """Clasifica un string rapidamente, no seleccionar los strings antes de llamar a la función impactará de manera negativa el output del pipeline"""
    if not text:
        return ([-1], 0)
    tokens = text.split(" ")
    semantic_classes: List[int] = []
    total_cuants = 0
    for t in tokens:
        t_class, t_cuant = classify_token(t)
        semantic_classes.append(t_class)
        total_cuants += t_cuant
    return (semantic_classes, total_cuants)