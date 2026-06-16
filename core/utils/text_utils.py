# PerfectOCR/core/utils/text_utils.py
import logging
import unicodedata
from typing import List, Tuple, Dict, Any, Optional
from core.utils.math_utils import text_encode
from core.utils.compiled_utils import validate_quant_chars, count_cuants
from core.utils.data_utils import VOWELS, REPLACEMENT_MAP
from core.utils.patterns import bad_title, numeric_fractions, has_digit_pattern, extension_suffix, correct_cuants, all_cuants, universal_money_regex, cant_frac_pattern, rfc_key_pattern, numeric_code, acronym_pattern, acromin_currency_pattern, cion_search_patt, suffix_pattern, cion_str, con_suffix_pattern, con_search_patt, con_str, umd_patterns, date_patterns, amount_fract, fraction_pattern, rfc_patterns, iva_patterns, phone_number, mail_pattern, cp_pattern, quant_runs_patterns, valid_cuant_pattern, monetary_pattern, clean_currency, edge_punt_pattern, hour_pattern, punt_split_pattern, sequence_middle_pattern, secuence_pattern, labels_pattern, size_pattern, id_prov_pattern, los_str, los_search_patt, los_suffix_pattern, swap_term_cuant

logger = logging.getLogger(__name__)

density_thr = (23.7, 103.7)
morph_thr = (-0.297, 0.337)

_bad_title = bad_title
_numeric_fractions = numeric_fractions
_has_digit_pattern = has_digit_pattern
_extension_suffix = extension_suffix
_correct_cuants = correct_cuants
_all_cuants = all_cuants
_universal_money_regex = universal_money_regex
_cant_frac_pattern = cant_frac_pattern
_swap_term_cuant = swap_term_cuant
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
_amount_fract = amount_fract
_fraction_pattern = fraction_pattern
_rfc_patterns = rfc_patterns
_iva_patterns = iva_patterns
_phone_number = phone_number
_mail_pattern = mail_pattern
_cp_pattern = cp_pattern
_quant_runs_patterns = quant_runs_patterns
_valid_cuant_pattern = valid_cuant_pattern
_monetary_pattern = monetary_pattern
_clean_currency = clean_currency
_edge_punt_pattern = edge_punt_pattern
_hour_pattern = hour_pattern
_punt_split_pattern = punt_split_pattern
_sequence_middle_pattern = sequence_middle_pattern
_secuence_pattern = secuence_pattern
_labels_pattern = labels_pattern
_size_pattern = size_pattern
_id_prov_pattern = id_prov_pattern

_replacement_map = REPLACEMENT_MAP
vowels = VOWELS

def normalice_text(s: str, hard_norm: bool) -> str:
    """"Normaliza texto eliminando apóstrofes, tildes, diéresis. NO ELIMINA CARACTERES DE NINGÚN TIPO, MISMO LEN() EN INPUT Y OUTPUT"""
    if not s:
        return ""

    norm_text = "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")
    if bool(_bad_title.fullmatch(norm_text)):
        norm_text = "".join(norm_text.split(" "))

    if norm_text and hard_norm:
        hard_text =  unicodedata.normalize('NFKD', norm_text).encode('ascii', 'ignore').decode('utf-8')
        return norm_text if not hard_text else hard_text
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
    elif bool(_swap_term_cuant.fullmatch(text)):
        return text.replace("$", "S")
    else:
        return text

def contains_umd(text: str) -> bool:
    """Valida si un string contiene UMD"""
    return bool(_umd_patterns.search(text))

def find_umd(text: str) -> str:
    """Aísla UMD/fracciones pegadas a texto y corrige fracciones OCR como C710 -> C/10 o 172 -> 1/2."""
    if not text:
        return ""

    text = text.strip()
    word_len = len(text)
    if word_len < 2:
        return text

    elif text.isalpha():
        return text

    elif text.isdecimal():
        return text

    elif bool(_phone_number.search(text)):
        return text

    elif bool(_date_patterns.search(text)):
        return text

    elif bool(_numeric_code.search(text)):
        return text

    result_parts: List[str] = []

    for word in text.split(" "):
        if not word:
            continue

        if word.isalpha():
            result_parts.append(word)
            continue

        matches = list(_umd_patterns.finditer(word))
        if not matches:
            result_parts.append(word)
            continue

        result = word

        for m in reversed(matches):
            raw_tok = m.group(0).strip()
            start, end = m.span()
            tok = raw_tok

            if _amount_fract.fullmatch(tok):
                tok = _correct_numbers(tok)
                dash_ind = tok.find("7")
                tok = tok[(dash_ind + 1):] if dash_ind >= 0 else tok
                tok = ("C/" + tok).strip()

            elif _numeric_fractions.fullmatch(tok):
                tok = _correct_numbers(tok)
                tok = tok.replace("7", "/", 1).strip()

            elif bool(_umd_patterns.fullmatch(tok)):
                tok = tok.strip()

            left_char = result[start - 1] if start > 0 else ""
            right_char = result[end] if end < len(result) else ""

            needs_left_space = bool(left_char) and left_char != " "
            needs_right_space = bool(right_char) and right_char != " "

            left_part = result[:start]
            right_part = result[end:]
            mid = f"{' ' if needs_left_space else ''}{tok}{' ' if needs_right_space else ''}"

            result = left_part + mid + right_part

        result_parts.append(result.strip())

    return space_removal(" ".join(result_parts).strip())

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

def validate_quant_pattern(text: str) -> bool:
    """Verifica si hay match completo de un patrón"""
    return bool(_quant_runs_patterns.fullmatch(text) or _universal_money_regex.fullmatch(text))
    
def is_quantitative(text: str) -> bool:
    """Válida rapidamente si un string es cuantitativo"""
    return False if len(text) < 3 else bool(validate_quant_pattern(text) or validate_quant_chars(text))

def contains_quantitative(text: str) -> bool:
    """Devuelve True si encuentra algún sub-string cuantitativo en el texto."""
    match = _all_cuants.search(text)
    return False if 3 > len(text) else bool(match and is_quantitative(match.group(0)))

def get_cuants(text: str) -> str:
    """Aísla cuantitativos SÓLO si están pegados a otros caracteres (ruido o texto). Si ya están separados por espacios, no modifica el texto."""
    if len(text) < 3 or text.isdecimal() or text.isalpha():
        return text

    words = text.strip().split(" ")
    result_parts: List[str] = []
    for word in words:

        if word.isalpha():
            result_parts.append(word)
            continue

        elif not bool(_has_digit_pattern.search(word)):
            result_parts.append(word)
            continue

        elif bool(_swap_term_cuant.fullmatch(word)):
            result_parts.append(word)
            continue

        if bool(_valid_cuant_pattern.search(word)):
            word = _valid_cuant_pattern.sub("5", word)
            word = ("$" + word[1:].strip())
        else:
            word = word

        matches = list(_all_cuants.finditer(word))
        if not matches:
            result_parts.append(word)
            continue

        result = word
        for m in reversed(matches):
            tok = m.group(0).strip()
            start, end = m.span()

            if not validate_quant_chars(tok) and bool(_monetary_pattern.search(tok)):
                quan_idx = tok.find("$")
                ctok = tok[(quan_idx + 1):].strip() if quan_idx >= 0 else tok[1:]
                ctok = ("$" + _correct_numbers(ctok))
                tok = ctok

            else:
                tok = tok
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
        continue

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
    if text.isalpha():
        logger.debug(f"TEXTO ALFABÉTICO NO SE FORMATEA: '{text}'")
        return text
    elif text.isdecimal():
        return (text + ".00").strip()
    text_0 = _clean_currency.sub("", text).strip()
    cuant_txt = _correct_numbers(text_0)
    if not cuant_txt.replace(".", "").isdecimal() or len(cuant_txt) > len(text):
        # logger.warning(f"CARACTER INTRUSO EN '{text}' NO SE PUDO FORMATEAR")
        return text
    else:
        return cuant_txt
            
def punct_strip(text: str) -> str:
    """Elimina los caracteres de puntuación que se encuentran al inicio y al final de cada token en el texto."""
    if not text:
        return ""
    if validate_quant_chars(text) or is_acronym(text):
        return text.strip()
    
    return _edge_punt_pattern.sub("", text).strip()

def separate_punt(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    
    if is_acronym(text):
        return text

    tokens = text.split(" ")
    processed_token_cuants: List[str] = []
    for t in tokens:
        # Mantiene intactas horas y cuantitativos puros; limpia los tokens mixtos.
        if not bool(_hour_pattern.search(t)) and not is_quantitative(t):
            cleaned_token_cuant = _punt_split_pattern.sub(" ", t)
            processed_token_cuants.append(cleaned_token_cuant)
        else:
            # Si es una hora, se mantiene intacta
            processed_token_cuants.append(t)
    # Une los tokens y usa space_removal para normalizar todos los espacios
    return space_removal(" ".join(processed_token_cuants))

def space_removal(text: str) -> str:
    """Normaliza espacios múltiples y limpia bordes."""
    if not text:
        return ""
    if " " not in text:
        return text
    if "  " not in text:
        return text.strip()
    return " ".join(text.split(" "))

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
        total_token_cuants = len(tokens) # Cantidad de tokens

        if total_token_cuants == 1:
            t_class, t_cuant = classify_token_cuant(tokens[0])
            final_results[pid] = ([t_class], t_cuant)
    
        else:
            token_classes: List[int] = []
            poly_total_cuant = 0
            for t in tokens:

                t_class, t_cuant = classify_token_cuant(t)
                token_classes.append(t_class)
                poly_total_cuant += t_cuant
                
            final_results[pid] = (token_classes, poly_total_cuant)
            continue

    return final_results

def classify_token_cuant(s: str) -> Tuple[int, int]:
    total_text = len(s)
    total_cuant = count_cuants(s)
    if not s:
        #  logger.info(f"VACIO 3RO: {s}")
        return (-1, 0)
        
    elif not any(c.isalnum() for c in s):
        #  logger.info(f"INVÁLIDO 1: '{s}'")
        return (-1, 0)
        
    elif s.isalpha():
        if bool(_size_pattern.search(s)):
            return (2, 0)
        #  logger.info(f"ALPHA 1ro: '{s}'")
        else:
            return (1, 0)
    
    elif s.isdecimal():
        if bool(_numeric_code.fullmatch(s)):
            return (3, total_cuant)
        else:
            return (5, total_text)
    
    if total_cuant == 0:
        if not any(c.isalpha() for c in s):
            return (-1, 0)
        
        elif bool(_fraction_pattern.search(s)):
            # logger.info(f"UMD por regex: '{s}'")
            return (2, 0)

        elif is_acronym(s):
            if bool(_acromin_currency_pattern.search(s)):
                # logger.info(f"ACRONIMO DEC: '{s}'")
                return (0, 0)
            else:
                return (1, 0)

        if not any(c in vowels for c in s):
            if _cant_frac_pattern.search(s):
                return (2, 0)

            elif 2 < total_text:
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
                # logger.info(f"CODE por inicio 0: '{s}'")
                return (3, total_cuant)
                
            elif validate_quant_pattern(s):
                # logger.info(f"CUANT por inicio 0: '{s}'")
                return (4, total_cuant)
            else:
                # logger.info(f"UMD por inicio 0: '{s}'")
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
    
    # logger.info(fr"REBELDES: '{s}'")
    dense_mean, morphology_mean = text_encode(s.lower())
    if dense_mean < density_thr[1] and morphology_mean > morph_thr[0]:
        # logger.info(f"CODE por codificacion: '{s}'")
        return (3, total_cuant)
    
    elif dense_mean > density_thr[1]:
        return (1, 0)

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
        return (1, 0)

    if not any(c in vowels for c in s) and total_text > 2:
        # logger.info(f"CODE por FALLBACK: '{s}'")
        return (3, total_cuant)
        
    # logger.info(f"Poligono sin clasificación, será descriptiva: '{s}'")
    return (1, 0)

def get_ids(id_registro: str, id_need: str):
    """
    Extrae datos relevantes en el ID Registro
    prov: id proveedor
    name: nombre de la imagen sin extensión
    """
    try:
        if id_need == "prov":
            matches = list(_id_prov_pattern.finditer(id_registro.strip()))
            return [match.group() for match in matches if _id_prov_pattern.fullmatch(match.group())][0]
        elif id_need =="name":
            return _extension_suffix.sub("", id_registro)
        else:
            return id_registro
    except ValueError as e:
        logger.info(f"NO se encontró id requerido: {e}")
    return id_registro

def format_elapsed_time(seconds: float) -> str:
    """Convierte segundos a formato HH:MM:SS.ms"""
    if seconds < 60.0:
        return f"{seconds:.6f}'s"
    minutes = int((seconds % 3600) // 60)
    if minutes < 60:
        return f"{minutes:02d}:M {seconds % 60:06.3f}'s"
    else:
        return f"{int(seconds // 3600):02d}:H {minutes:02d}:M {seconds % 60:06.3f}'s"

def fast_classfier(text: str) -> Tuple[List[int], int]:
    """Clasifica un string rapidamente, no seleccionar los strings antes de llamar a la función impactará de manera negativa el output del pipeline"""
    if not text:
        return ([-1], 0)
    total_cuants = 0
    semantic_classes: List[int] = []
    tokens = text.split(" ")
    for t in tokens:
        t_class, t_cuant = classify_token_cuant(t)
        semantic_classes.append(t_class)
        total_cuants += t_cuant
    return (semantic_classes, total_cuants)

def _correct_numbers(text: str) -> str:
    """Solo corrije numeros, no borra caracteres"""
    return text if text.isdecimal() else _correct_cuants.sub(lambda match: _replacement_map[match.lastgroup], text)

def noramalice_df(text: str, separator: str) -> str:
    text = normalice_text(text, True)
    text = text.replace(separator, " ")
    text = space_removal(text)
    return (text + separator)

def its_similar(word: str, suspect: str) -> bool:
    suspect_slice = suspect[:len(word)]
    if word == suspect_slice:
        return True
    else:
        suspect_slice = _correct_numbers(suspect_slice)
        return suspect_slice.isdecimal() and word == suspect_slice

# def classify_special(s: str) -> Tuple[str, Optional[int]]:
#     date_matches = list(_date_patterns.finditer(s))
#     if date_matches:
#         date = [match.group() for match in date_matches if _date_patterns.fullmatch(match.group())]
#         return ("", None) if not date else (date[0], 9)
#
#     rfc_matches = list(_rfc_key_pattern.finditer(s))
#     if rfc_matches:
#         rfc = [match.group() for match in rfc_matches if len(match.group()) > 10]
#         return ("", None) if not rfc else (rfc[0], 7)
#
#     phone_matches = list(_phone_number.finditer(s))
#     if phone_matches:
#         phone = [match.group() for match in phone_matches if len(match.group()) > 8]
#         return ("", None) if not phone else (phone[0], 10)
#
#     mail_matches = list(_mail_pattern.finditer(s))
#     if mail_matches:
#         mail = [match.group() for match in mail_matches if match.group()]
#         return ("", None) if not mail else (mail[0], 11)
#
#     iva_matches = list(_iva_patterns.finditer(s))
#     if iva_matches:
#         iva = [match.group() for match in iva_matches if validate_quant_chars(match.group()) or not match.group().isalpha()]
#         return ("", None) if not iva else (iva[0], 8)
#
#     cp_matches = list(_cp_pattern_patterns.finditer(s))
#     if cp_matches:
#         cp = [match.group() for match in cp_matches if _cp_pattern.fullmatch(match.group())]
#         return ("", None) if not cp else (cp[0], 12)
#     return ("", None)
#
#
# polygons: Dict[str, Polygons] = manager.workflow.polygons
#         final_polygons: Dict[str, Dict[str, Any]] = {}
#         final_results: Dict[str, List[int]] = {}
#         for poly_id, poly_data in polygons.items():
#             text = poly_data.ocr_text or ""
#             kf_text, kf = classify_special(text)
#             if not kf_text or kf is None:
#                 final_polygons[poly_id] = {"text": text}
#                 continue
#
#             final_results[poly_id] = [kf]
#
#             kf_match = kf_text  # el que devuelve classify_special()
#             kf_text = kf_match.replace(" ", "").strip()
#
#             if text == kf_match:
#                 text_recon = kf_text
#             else:
#                 index_kf = text.find(kf_match)
#                 if index_kf >= 0:
#                     end_kf = index_kf + len(kf_match)
#                     text_recon = (text[:index_kf] + " " + kf_text + " " + text[end_kf:])
#                     text_recon = " ".join(text_recon.split())
#                 else:
#                     text_recon = text
#
#             text_list = text_recon.split(" ")
#             idx_kf = text_list.index(kf_text)
#             s_class, t_cuant = fast_classfier(text_recon)
#
#             s_classiy: List[int] = []
#             for i, s_c in enumerate(s_class):
#                 if i == idx_kf:
#                     s_classiy.append(0)
#                 else:
#                     s_classiy.append(s_c)
#
#             s_class = s_classiy
#             final_polygons[poly_id] = {"text": text_recon, "sc": s_class, "cuant_chars": 0}
#             continue
#
#         worker_name = context.get("worker_name") or "text_refiner"
#         manager.update_ocr_results(final_polygons, worker_name)
#         manager.update_key_field(final_results)
