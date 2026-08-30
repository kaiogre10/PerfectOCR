# PerfectOCR/core/utils/patterns.py
import regex as re
from typing import List, Pattern

_currency_variants = r"[$Ss]"
_punt_quant_chars = r'[.,]'
_diagonal_variants = r'[7\/]'
_digit_pattern = r"[0-9OQo]"
_base_date_num_str = r'[012OQo][0-9OQo]'
_month_name_str = r'(?:ene(?:ro)?|feb(?:rero)?|mar(?:zo)?|abr(?:il)?|may(?:o)?|jun(?:io)?|jul(?:io)?|ago(?:s(?:to)?)?|sep(?:t(?:iembre)?)?|oct(?:ubre)?|nov(?:iembre)?|dic(?:iembre)?)'
# _num_fract = r"[1-9]"
# _standar_size_den = r'(2|4|8|16|32)'
_c_variants = r"[Cc]"
_cp_letters = r'(?:C\.?\s*P\.?|C\s+P|CP)'
_bic_variants = r'\b(B(1C|lC|\|C|¡C|!C))\b'
space_pattern: Pattern[bytes] = re.compile(rb"\s+", re.IGNORECASE)

_digits_base = r'0123456789'
_zero_base = r'OQoD'
_one_base = r'|liI!¡'
_two_base = r'Zz?'
_three_base = r'3'
_four_base = r'A'
_five_base = r'$Ss'
_six_base = r'G'
_seven_base = r'/'
_eight_base = r'B'
_nine_base = r'qg'

_all_zeros = rf"[0{_zero_base}]"
_extended_digits = rf"[{_digits_base}{_zero_base}{_one_base}{_two_base}{_four_base}{_five_base}{_six_base}{_seven_base}{_eight_base}{_nine_base}]"   # [0-9OQoD|liI!¡Zz?A$SsG/Bqg]

cleaner_pattern = re.compile(r"[^a-zA-Z\s]", re.IGNORECASE)
float_time = re.compile(r"Tiempo:\s*(\d+\.\d)\b")
_stick_chars = r'[!1lIi¡]'
_stick_set = _stick_chars[1:-1]
_l_variants = rf"[L{_stick_set}]"
_i_variants = rf"[{_stick_set}]"
_o_variants = r"[Oo0Q]"
_n_variants = r"[Nn]"
_s_variants = rf'[5{_five_base}]'

cion_str = r"CION"
con_str = r"CON"
los_str = r'LOS'

cion_search_patt = re.compile(rf"(?<=[A-Za-z]){cion_str}(?:\b|$|\s)", re.IGNORECASE)
con_search_patt = re.compile(rf"(?:^{con_str}$|(?<=[A-Za-z]){con_str}(?:\b|$|\s))", re.IGNORECASE)
los_search_patt = re.compile(rf"(?:^{los_str}$|(?<=[A-Za-z]){los_str}(?:\b|$|\s))", re.IGNORECASE)

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

_los_typos_regex = (
    rf"(?:"
    rf"^{_l_variants}\s*{_o_variants}\s*{_s_variants}$"
    rf"|(?<=[A-Za-z]){_l_variants}\s*{_o_variants}\s*{_s_variants}(?:\b|$|\s)"
    rf")"
)
los_suffix_pattern = re.compile(_los_typos_regex)

extension_suffix = re.compile(r'\.([A-Za-z]+)$', re.IGNORECASE)
has_digit_pattern = re.compile(r"\d", re.IGNORECASE)
swap_term_cuant = re.compile(r'^[A-Za-z]+\$$')
# Patrón super estricto para identificar "BIC" y variantes OCR ("B1C", "BlC", "B|C", "B¡C", "B!C", "BIC", pero SOLO esas, sin prefijos ni sufijos)
# _dixon_variants = rf'(D(1X|LX{_zeros_to_sub}N'
labels_pattern: Pattern[str] = re.compile(_bic_variants, re.IGNORECASE)

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
_acronim = r'(?:[A-Za-z]\.)+[A-Za-z]\.?'
acromin_currency_pattern = re.compile(r"(?:\d[\d,.]*)?\s*m\s*\.?\s*n\.?", re.IGNORECASE)

acronym_pattern: Pattern[str] = re.compile(rf'^({_acronim}|sa|cv|mn)[:;,.]?$', re.IGNORECASE)
bad_title: Pattern[str] = re.compile(r'^\S(?:\s+\S)+\s*$', re.IGNORECASE)

# Datos Globales
_phone_str = r'^(cel|tel)((?:\:|\s))'
phone_number = re.compile(rf"(((?:{_phone_str}){_digit_pattern}{{10}})|(^{_digit_pattern}{{10}}))(?:$|(?=\s))", re.IGNORECASE)
mail_pattern = re.compile(r'\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.?(?:com|[a-zA-Z0-9]+)\b', re.IGNORECASE)
cp_pattern: Pattern[str] = re.compile(rf'\b{_cp_letters}\s*{_digit_pattern}{{5}}\b')
numeric_code: Pattern[str] = re.compile(rf"(?<=\s){_all_zeros}{_digit_pattern}+$")

_rfc_acronyms: Pattern[str] = re.compile(r'\b(R\.?F\.?C\.?)\b', re.IGNORECASE)
rfc_key_pattern: Pattern[str] = re.compile(r'([A-Z]{3,4}\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])[A-Z0-9]{3})', re.IGNORECASE)

rfc_patterns: Pattern[str] = re.compile("|".join(p.pattern for p in [rfc_key_pattern, _rfc_acronyms]), re.IGNORECASE)

iva_patterns: Pattern[str] = re.compile(r'\b(I\.?V\.?A\.?)\b', re.IGNORECASE)

_date_patterns_list: List[Pattern[str]] = [
    # Día + mes en letras + año en un solo string OCR (ej. "21 mar 2023")
    re.compile(rf'\b(?:{_month_name_str}|{_base_date_num_str}\s+{_month_name_str}(?:\s*\.?\s*(?:19\d{{2}}|20\d{{2}}|\d{{2}}))?)\b'),
    # Fechas completas y día/mes: Usa el bloque base para evitar confundirse con fracciones/cuantitativos
    re.compile(rf'\b{_base_date_num_str}{_diagonal_variants}{_base_date_num_str}(?:{_diagonal_variants}(?:\d{{2,4}}))(?=(\b|$))', re.IGNORECASE),
    # Años
    re.compile(r'\b(199\d|20\d{2})\b', re.IGNORECASE),
    re.compile(rf'\b(?:199[0-9{_all_zeros}]|2[0{_all_zeros}][0-9{_all_zeros}]{{2}})\b',re.IGNORECASE)
]
date_patterns = re.compile("|".join(p.pattern for p in _date_patterns_list), re.IGNORECASE)
mont_pattern = re.compile(rf"\b{_month_name_str}(?=(\b|$))")

_unities = re.compile(r'(g|l|m)', re.IGNORECASE)
_mass_pattern = re.compile(r'(kg|gr|grs|mg|lb|lbs|oz|ton)', re.IGNORECASE)
_vol_pattern = re.compile(r'(lt|ltr|ltrs|lts|ml|gal)', re.IGNORECASE)
_len_pattern = re.compile(r'(mm|cm|km|in|ft|mts|m2|cm2|km2)', re.IGNORECASE)
size_pattern = re.compile(r'(gde|med|ch|paq)', re.IGNORECASE)
measure_unities = re.compile("|".join(p.pattern for p in [size_pattern, _mass_pattern, _vol_pattern, _len_pattern]))

# Detecta expresiones como "C / 3 + 2" o variantes OCR para fracciones extendidas con suma: letra C, barra, número, signo más, número.
_extended_fraction_pattern = re.compile(r'(?<!\d)[Cc]\s*/\s*\d+\s*\+\s*\d+(?![A-Za-z0-9])')

# Detecta fracciones estándar como "C / 3" o variantes OCR: letra C seguida de barra y número (sin suma)
_full_fraction_pattern = re.compile(r'(?<!\d)[Cc]\s*/\s*\d+\b')

# Detecta expresiones de tipo fracción incompleta como "/3", es decir, solo una barra y número, sin prefijo
_semifraction_pattern = re.compile(r'(?<![A-Za-z0-9])/\d+\b')

# Detecta fracciones de tipo "C /", "C / 3", permitiendo que el número sea opcional (ej. "C / " o "C / 5")
_cuant_frac_str = rf'{_c_variants}\s*{_diagonal_variants}'

semi_c_fraction = re.compile(rf'\b{_cuant_frac_str}(?:$|(?=\s))')
cant_frac_pattern = re.compile(rf'{_cuant_frac_str}\s*(?:\s|(?=|\d)(?=|\b))')

# Detecta fracciones con guión como "C-3", es decir, letra C, guión y número, sin prefijo.
c_dash_fraction_pattern = re.compile(r'(?<![A-Za-z0-9])[Cc]-\d+\b')

# Detecta fracciones "naturales" tipo "1/2", "1/25", donde ambos lados (numerador y denominador) pueden tener de 1 a 3 dígitos.
numeric_fractions = re.compile(rf'(?<!\d)(\b{_digit_pattern}{{1,2}}{_diagonal_variants}{_digit_pattern}{{1,3}})(?:$|(?=\s))')
measure_fractions = re.compile(rf'(?<!\d)(?<!\d{{2}})\s*{_diagonal_variants}\s*[1-9]\d{{1,3}}(?:$|(?=\s))')

# Detecta cantidades con fracción que usan variantes de "C/" para unidades ("C/10", permitiendo variantes OCR en el carácter de barra y ceros sustituidos
amount_fract = re.compile(
    rf"(?<!\b){_c_variants}\s*{_diagonal_variants}\s*{_digit_pattern}+(?=\b|$)|"
    rf"\b{_c_variants}{_diagonal_variants}\s*{_digit_pattern}+\b"
)

fraction_pattern = re.compile("|".join(p.pattern for p in [_extended_fraction_pattern, _full_fraction_pattern, c_dash_fraction_pattern, amount_fract, measure_fractions, numeric_fractions, semi_c_fraction, _semifraction_pattern]), re.IGNORECASE)

_umd_paterns_list: List[Pattern[str]] = [
    re.compile(r'(?<![A-Za-z0-9])\d{1,4}\s*m(?:l|1|\||!)(?=$|[^A-Za-z0-9])', re.IGNORECASE),
    re.compile(rf'\b\d+\s*(?:{_unities.pattern})\b', re.IGNORECASE),
    re.compile(rf'((?<!\w)(?:\d+\s*)?(?:{measure_unities.pattern}))\b', re.IGNORECASE),
    re.compile(r'\b[1-9]\d{0,2}\s*/\s*[1-9]\d{0,2}\b', re.IGNORECASE),
    re.compile(r'\b\d+(?:\s*[xX]\s*\d+)+\b', re.IGNORECASE), # Dimensiones (10x20)
    re.compile(r'#\s*\d+'),
    re.compile(r'\b\d{2,3}H(?=\s|$)', re.IGNORECASE)
]

_umd_generals = re.compile("|".join(p.pattern for p in _umd_paterns_list), re.IGNORECASE)
umd_patterns = re.compile("|".join(p.pattern for p in [fraction_pattern, _umd_generals]), re.IGNORECASE)

_clean_currency_str = rf"^(?:{_currency_variants})|,"
clean_currency = re.compile(_clean_currency_str)

monetary_pattern = re.compile(rf'^{_currency_variants}\s*\d+(?:{_punt_quant_chars}\d+)?(?=\b|\s|$)')
_zeros_to_compile = (rf"([{_zero_base}])")
zeros_variants = re.compile(_zeros_to_compile)

valid_cuant_pattern = re.compile(r'(?<=\$)\$|(?<=\$\d)\$|(?<=\$\d\d)\$', re.IGNORECASE)

# Detecta el "cuerpo" de un monto numérico, es decir la parte de los dígitos principales de una cantidad monetaria o cuantitativa.
_amount_body_pattern = (
    rf"(?:{_digit_pattern}+(?:{_punt_quant_chars}{_digit_pattern}+)?|"
    rf"{_digit_pattern}{{1,3}}(?:{_punt_quant_chars}{_digit_pattern}{{3}})*)(?:{_punt_quant_chars}{_digit_pattern}{{2}})"
)

_token_pattern = (
    rf"{_currency_variants}\s*{_amount_body_pattern}|"
    rf"{_amount_body_pattern}\s*{_currency_variants}|"
    rf"{_amount_body_pattern}"
)
# Detecta patrones cuantitativos en texto:
_token_cuant = re.compile(_token_pattern)

# Patrón: Montos con símbolo al inicio ($ 80.50)
_start_pattern = rf"^{_currency_variants}\s*{_amount_body_pattern}$"
_start = re.compile(_start_pattern, re.IGNORECASE)

# Patrón: Monto con símbolo en medio (80 $ 50)
_middle_pattern = rf"^{_amount_body_pattern}\s*{_currency_variants}\s*{_amount_body_pattern}$"
_middle = re.compile(_middle_pattern, re.IGNORECASE)

# Patrón: Múltiples montos seguidos de símbolo ($100 $200)
_multi_pattern = rf"^(?:\s*{_currency_variants}\s*{_amount_body_pattern}\s*){{2,}}$"
_multi = re.compile(_multi_pattern, re.IGNORECASE)

# Patrón: Decimales grandes tipo 1,230.50 (sin $)
_decimal = re.compile(rf"^\d{{1,3}}(?:{_punt_quant_chars}\d{{3}})*{_punt_quant_chars}\d{{2,}}$")

quant_runs_patterns = re.compile("|".join(p.pattern for p in [_decimal, _start, _middle, _multi]), re.IGNORECASE)

# 1. Módulo de Divisa: Valida el prefijo monetario y espacios opcionales subsecuentes.
_currency_prefix = rf"(?:{_currency_variants}\s*)?"

# 2. Módulo Entero: 1-3 dígitos, opcionalmente seguidos por bloques repetidos de separador + 3 dígitos.
_integer_block = rf"{_extended_digits}{{1,3}}(?:\s*{_punt_quant_chars}\s*{_extended_digits}{{3}})*"

# 3. Módulo Decimal: separador decimal obligatorio + exactamente 2 dígitos.
_decimal_block = rf"\s*{_punt_quant_chars}\s*{_extended_digits}{{2}}"

# 4. Patrón universal: moneda opcional al inicio + entero + decimal obligatorio.
_universal_money_pattern = (
    rf"(?<!{_extended_digits})"
    rf"{_currency_prefix}"
    rf"{_integer_block}"
    rf"{_decimal_block}"
    rf"(?!{_extended_digits})"
)
universal_money_regex = re.compile(_universal_money_pattern)
all_cuants = re.compile("|".join(p.pattern for p in [quant_runs_patterns, _token_cuant, universal_money_regex]))

_correct_ocr = (
    rf"(?P<zero>[{_zero_base}])|"
    rf"(?P<one>[{_one_base}])|"
    rf"(?P<two>[{_two_base}])|"
    rf"(?P<three>[{_three_base}])|"
    rf"(?P<four>[{_four_base}])|"
    rf"(?P<five>[{_five_base}])|"
    rf"(?P<six>[{_six_base}])|"
    rf"(?P<seven>[{_seven_base}])|"
    rf"(?P<eight>[{_eight_base}])|"
    rf"(?P<nine>[{_nine_base}])"
)
correct_cuants = re.compile(_correct_ocr)

# paddle_silene = re.compile(r".*OMP_NUM_THREADS.*|.*PLEASE USE OMP_NUM_THREADS WISELY.*")