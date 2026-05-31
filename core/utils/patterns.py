# PerfectOCR/core/utils/patterns.py
import re
from typing import List, Pattern

_zeros_str = r'00'
_currency_pattern = r"[$]"
_punt_quant_chars = r'[.,]'
_zeros_to_sub = r'[OQDo]'
_monetary_to_sub =  r'^[Ss]\d'
_semi_zeros = r'[OQDo]{2}'
_digit_pattern = r"[0-9oOQD]"
_clean_currency_str = r"^(?:\$)|,"
_base_date_num_str = r'[0123O][0-9O]'
_valid_cuant_str = r'(?<=\$)\$|(?<=\$\d)\$|(?<=\$\d\d)\$'
_month_name_str = r'(?:ene(?:ro)?|feb(?:rero)?|mar(?:zo)?|abr(?:il)?|may(?:o)?|jun(?:io)?|jul(?:io)?|ago(?:s(?:to)?)?|sep(?:t(?:iembre)?)?|oct(?:ubre)?|nov(?:iembre)?|dic(?:iembre)?)\.?'
_c_variants = r"[C(c]"
_i_variants = r"[iI|!1¡]"
_n_variants = r"[nN]"
_o_variants = r"[Oo0Q]"
cion_str = r"CION"
con_str = r"CON"
cion_search_patt = re.compile(rf"(?<=[A-Za-z]){cion_str}(?:\b|$|\s)", re.IGNORECASE)
con_search_patt = re.compile(rf"(?:^{con_str}$|(?<=[A-Za-z]){con_str}(?:\b|$|\s))", re.IGNORECASE)
# Solo puede aparecer después de algún otro caracter
_cion_typos_regex = (
    rf"(?<=[A-Za-z])"                   # Debe aparecer después de caracteres alfabéticos
    rf"{_c_variants}\s*"
    rf"{_i_variants}\s*"
    rf"{_o_variants}\s*"
    rf"{_n_variants}"
    rf"(?:\b|$|\s)"                     # Al final del string, antes de un espacio, o en medio (palabra terminada)
)
suffix_pattern = re.compile(_cion_typos_regex)
_con_typos_regex = (
    rf"(?:"
    rf"^{_c_variants}\s*{_o_variants}\s*{_n_variants}$"
    rf"|(?<=[A-Za-z]){_c_variants}\s*{_o_variants}\s*{_n_variants}(?:\b|$|\s)"
    rf")"
)
con_suffix_pattern = re.compile(_con_typos_regex)

semi_zeros_pattern = re.compile(_semi_zeros)
zeros_pattern = re.compile(_zeros_to_sub)
monetary_pattern = re.compile(_monetary_to_sub)
valid_cuant_pattern = re.compile(_valid_cuant_str, re.IGNORECASE)
# _punt_pattern = re.compile(_punt_quant_chars, re.IGNORECASE)

# Patrón super estricto para identificar "BIC" y variantes OCR ("B1C", "BlC", "B|C", "B¡C", "B!C", "BIC", pero SOLO esas, sin prefijos ni sufijos)
_bic_variants = r'^(B(1C|lC|\|C|¡C|!C))$'
# _dixon_variants = rf'(D(1X|LX{_zeros_to_sub}N'
labels_pattern: Pattern[str] = re.compile(_bic_variants)

# Patrón que extrae los IDS del documento
id_prov_pattern = re.compile(r'^[A-Z]{4}')

# Patrón para secuencias especiales de 2 o más caracteres no alfanuméricos (excluyendo espacio, $, /,)
secuence_pattern: Pattern[str] = re.compile(r'[^a-zA-Z0-9\s/$]{2,}', re.IGNORECASE)
sequence_middle_pattern: Pattern[str] = re.compile(r'(?<=[a-zA-Z0-9$/])[^a-zA-Z0-9\s$/]{2,}(?=[a-zA-Z0-9$/])', re.IGNORECASE)

hour_pattern: Pattern[str] = re.compile(rf'\b{_base_date_num_str}:[0-5O][0-9O](?::[0-5O][0-9O])?\b', re.IGNORECASE)
punt_split_pattern: Pattern[str] = re.compile(r"[*_'=.,:;&]", re.IGNORECASE)
_edge_chars = punt_split_pattern.pattern[:-1] + r"\-]"
edge_punt_pattern = re.compile(rf'^({_edge_chars}+)|({_edge_chars}+)$', re.IGNORECASE)

# Siglas/Acrónimos
# Ahora el patrón fuerza secuencias de letra.punto repetidas (ej: P.U.C.D. etc). El último . es opcional solo después de la última letra.
_acronim = r'(?:[A-Za-z]\.)+[A-Za-z]\.?'
acromin_currency_pattern = re.compile(r"(?:\d[\d,.]*)?\s*m\s*\.?\s*n\.?", re.IGNORECASE)

acronym_pattern: Pattern[str] = re.compile(rf'^({_acronim}|sa|cv|mn)[:;,.]?$', re.IGNORECASE)
# _bad_title: Pattern[str] = re.compile(r'^([A-Za-z0-9])(?: [A-Za-z0-9])+$', re.IGNORECASE)

# Datos Globales
_cp_letters = r'(?:C\.?\s*P\.?|C\s+P|CP)'
phone_number = re.compile(r'(?:\b(?:cel|tel)\b[\s:,-]*)?(?:\d[\s\-]*){10}\b', re.IGNORECASE)

mail_pattern: Pattern[str] = re.compile(r'\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b', re.IGNORECASE)
cp_pattern: Pattern[str] = re.compile(rf'\b{_cp_letters}\s*{_digit_pattern}{{5}}\b', re.IGNORECASE)
numeric_code: Pattern[str] = re.compile(rf'^{_zeros_str}[0-9]+$', re.IGNORECASE)

# Fecha
_date_patterns_list: List[Pattern[str]] = [
    # Día + mes en letras + año en un solo string OCR (ej. "21 mar 2023")
    re.compile(rf'\b(?:{_month_name_str}|{_base_date_num_str}\s+{_month_name_str}(?:\s*\.?\s*(?:19\d{{2}}|20\d{{2}}|\d{{2}}))?)\b', re.IGNORECASE),
    # Fechas completas y día/mes: Usa el bloque base para evitar confundirse con fracciones/cuantitativos
    re.compile(rf'\b{_base_date_num_str}[\s\/\-]{_base_date_num_str}(?:[\s\/\-](?:\d{{2,4}}))?\b', re.IGNORECASE),
    # Años
    re.compile(r'\b(199\d|20\d{2})\b', re.IGNORECASE),
    re.compile(rf'\b(?:199[0-9{_zeros_to_sub[1:-1]}]|2[0{_zeros_to_sub[1:-1]}][0-9{_zeros_to_sub[1:-1]}]{{2}})\b',re.IGNORECASE)
]

_unities = re.compile(r'(g|l|m)', re.IGNORECASE)
_mass_pattern = re.compile(r'(kg|gr|grs|mg|lb|lbs|oz|ton)', re.IGNORECASE)
_vol_pattern = re.compile(r'(lt|ltr|ltrs|lts|ml|gal)', re.IGNORECASE)
_len_pattern = re.compile(r'(cm|mm|km|in|ft|mts?|m2|m\^2|m²|cm2|cm\^2|cm²|km2|km\^2|km²)', re.IGNORECASE)
size_pattern = re.compile(r'(gde|med|ch|paq)', re.IGNORECASE)
measure_unities = re.compile("|".join(p.pattern for p in [size_pattern, _mass_pattern, _vol_pattern, _len_pattern]), re.IGNORECASE)

_extended_fraction_pattern = re.compile(r'(?<!\d)[Cc]\s*/\s*\d+\s*\+\s*\d+(?![A-Za-z0-9])', re.IGNORECASE)
_full_fraction_pattern = re.compile(r'(?<!\d)[Cc]\s*/\s*\d+\b', re.IGNORECASE)
_semifraction_pattern = re.compile(r'(?<![A-Za-z0-9])/\d+\b', re.IGNORECASE)
semi_c_fraction = re.compile(r'(?<!\d)[Cc]\s*/(?:\s*\d+)?\b', re.IGNORECASE)
_c_dash_fraction_pattern = re.compile(r'(?<![A-Za-z0-9])[Cc]-\d+\b', re.IGNORECASE)

fraction_pattern = re.compile("|".join(p.pattern for p in [_extended_fraction_pattern, _full_fraction_pattern, semi_c_fraction, _semifraction_pattern, _c_dash_fraction_pattern]), re.IGNORECASE)

_umd_paterns_list: List[Pattern[str]] = [
    re.compile(r'(?<![A-Za-z0-9])\d{1,4}\s*m(?:l|1|\||!)(?=$|[^A-Za-z0-9])', re.IGNORECASE),
    re.compile(rf'\b\d+\s*(?:{_unities.pattern})\b', re.IGNORECASE),
    re.compile(rf'((?<!\w)(?:\d+s*)?(?:{measure_unities.pattern}))\b', re.IGNORECASE),
    re.compile(r'\b[1-9]\d{0,2}\s*/\s*[1-9]\d{0,2}\b', re.IGNORECASE),
    re.compile(r'\b\d+(?:\s*[xX]\s*\d+)+\b', re.IGNORECASE), # Dimensiones (10x20)
    re.compile(r'#\s*\d+')
]

_umd_generals = re.compile("|".join(p.pattern for p in _umd_paterns_list), re.IGNORECASE)
umd_patterns = re.compile("|".join(p.pattern for p in [fraction_pattern, _umd_generals]), re.IGNORECASE)

amount_fract = re.compile(fr"[Cc](?:[7/])\d{_zeros_to_sub}+", re.IGNORECASE)
umd_cor = re.compile(rf'\d{_zeros_to_sub}(?=\s|$)', re.IGNORECASE)

# _final_clean_currency = re.compile(r"[^0-9.]", re.IGNORECASE)
clean_currency = re.compile(_clean_currency_str, re.IGNORECASE)
# Patrón: S al inicio, al menos 3 dígitos entre la S y un punto o coma
# _s_correct_pattern = re.compile(r'^S\d{3,}[.,]', re.IGNORECASE)

# _currency_stick_pattern: Pattern[str] = re.compile(r'([a-zA-Z])([\$])')

# Usa los strings en las interpolaciones
_amount_body_pattern = (
    rf"(?:{_digit_pattern}+(?:{_punt_quant_chars}{_digit_pattern}+)?|" # Caso simple: 10.50
    rf"{_digit_pattern}{{1,3}}(?:{_punt_quant_chars}{_digit_pattern}{{3}})*)(?:{_punt_quant_chars}{_digit_pattern}{{2}})" # Caso miles: 1,000.00
)

_token_pattern = (
    rf"{_currency_pattern}\s*{_amount_body_pattern}|"
    rf"{_amount_body_pattern}\s*{_currency_pattern}|"
    rf"{_amount_body_pattern}"
)
# Detecta patrones cuantitativos en texto:
token = re.compile(_token_pattern, re.IGNORECASE)

# Patrón: Montos con símbolo al inicio ($ 80.50)
_start_pattern = rf"^{_currency_pattern}\s*{_amount_body_pattern}$"
_start = re.compile(_start_pattern, re.IGNORECASE)

# Patrón: Monto con símbolo en medio (80 $ 50)
_middle_pattern = rf"^{_amount_body_pattern}\s*{_currency_pattern}\s*{_amount_body_pattern}$"
_middle = re.compile(_middle_pattern, re.IGNORECASE)

# Patrón: Múltiples montos seguidos de símbolo ($100 $200)
_multi_pattern = rf"^(?:\s*{_currency_pattern}\s*{_amount_body_pattern}\s*){{2,}}$"
_multi = re.compile(_multi_pattern, re.IGNORECASE)

# Patrón: Decimales grandes tipo 1,230.50 (sin $)
_decimal = re.compile(rf"^\d{1,3}(?:{_punt_quant_chars}\d{3})*{_punt_quant_chars}\d{2,}$")

quant_runs_patterns = re.compile("|".join(p.pattern for p in [_decimal, _start, _middle, _multi]), re.IGNORECASE)

# Patrón: Monto terminado en símbolo (80.00 $)
# _end_pattern = rf"^{_amount_body_pattern}\s*{_currency_pattern}$"
# _end = re.compile(_end_pattern, re.IGNORECASE)

# Detecta terminaciones típicas de dinero (.00 ó ,00)
end_cuant_str = f'.{_zeros_str}'
# _end_quants = re.compile(rf'{_punt_quant_chars}{_zeros_str}$', re.IGNORECASE)

# Patrón equivalente a split, pero requiere $ al inicio y una cantidad
_split_pattern = rf"{_currency_pattern}\s*{_amount_body_pattern}"
split = re.compile(_split_pattern, re.IGNORECASE)

_rfc_acronyms: Pattern[str] = re.compile(r'\b(R\.?F\.?[C(]\.?)\b', re.IGNORECASE)
rfc_key_pattern: Pattern[str] = re.compile(r'([A-ZÑ]{3,4}\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])[A-Z0-9]{3})', re.IGNORECASE)

rfc_patterns: Pattern[str] = re.compile("|".join(p.pattern for p in [rfc_key_pattern, _rfc_acronyms]), re.IGNORECASE)

iva_patterns: Pattern[str] = re.compile(r'\b(I\.?V\.?A\.?)\b', re.IGNORECASE)
date_patterns = re.compile("|".join(p.pattern for p in _date_patterns_list), re.IGNORECASE)

__all__ = [
    "rfc_key_pattern",
    "numeric_code",
    "acronym_pattern",
    "acromin_currency_pattern",
    "cion_search_patt",
    "suffix_pattern",
    "cion_str",
    "con_suffix_pattern",
    "con_search_patt",
    "con_str",
    "umd_patterns",
    "date_patterns",
    "umd_cor",
    "amount_fract",
    "zeros_pattern",
    "fraction_pattern",
    "rfc_patterns",
    "iva_patterns",
    "phone_number",
    "mail_pattern",
    "cp_pattern",
    "quant_runs_patterns",
    "token",
    "valid_cuant_pattern",
    "split",
    "semi_zeros_pattern",
    "monetary_pattern",
    "end_cuant_str",
    "clean_currency",
    "edge_punt_pattern",
    "hour_pattern",
    "punt_split_pattern",
    "sequence_middle_pattern",
    "secuence_pattern",
    "labels_pattern",
    "size_pattern",
    "semi_c_fraction",
    "measure_unities",
    "id_prov_pattern",
]
