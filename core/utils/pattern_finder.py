import re
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

def is_acronym(text: str) -> bool:
        """
        Detecta siglas del tipo A.B.C. o P.U.C.D con punto final opcional
        y, a partir de ahora, admite un signo ':' ';' ',' opcional al final.
        """
        pattern = r'^([A-Za-z]\.){2,}[A-Za-z]\.?(?:[:;,])?$'
        return re.search(pattern, text.strip()) is not None

def find_umd(s: str) -> bool:
        try:
            if not s or not s.strip():
                return False

            # Regex Maestra para capturar la unidad y el valor
            # Nota: La diagonal debe ser escapada: \/
            umd_patterns = r'\b(\d+([,\.]\d+)?)\s*(/)?\s*(k(g(r)?|ilo(s)?)|g(r|ramo(s)?|m)?|mg|m(t(r)?|etro(s)?|2|\\^2)?|cm|l(t(r)?|itro(s)?|ml)?|cc|ud(s)?|pza(s)?|cj(s)?)\b'
            if re.search(umd_patterns, s):
                 return True
            return False

        except Exception as e:
            logger.warning(f"No se hallaron unidades de medida: {e}", exc_info=True)
            return False

def find_rfc(s: str) -> bool:
    try:
        if not s or not s.strip():
            return False
        rfc_code = r'^([A-ZÑ&]{3,4})\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])[A-Z0-9]{3}$'
        rfc_word = r'\b(R\.?F\.?C\.?)\b'

        if is_acronym(s):
            if re.search(rfc_word, s):
                return True
            else:
                if re.search(rfc_code, s):
                    return True

        return False
    except Exception as e:
        logger.warning(f"Error buscando RFC: {e}", exc_info=True)
        return False

def find_iva(s: str) -> bool:
    try:
        if not s or not s.strip():
            return False
        iva_word = r'\b(I\.?V\.?A\.?)\b'

        if is_acronym(s):
            if re.search(iva_word, s):
                return True
            else:
                return False

        return False
    except Exception as e:
        logger.warning(f"Error buscando IVA: {e}", exc_info=True)
        return False

def quantitative_runs(s: str) -> List[Tuple[int, int, str]]:
    s = (s or "").strip()
    if not s:
        return []
    currency = r"[$€£¥¢]"
    amount_body = r"(?:\d{1,3}(?:[.,]\d{3})+|\d+)(?:[.,]\d+)?"
    # cuantitativo: decimales o con símbolo (sin %)
    quant_token = rf"{currency}\s*{amount_body}|{amount_body}\s*{currency}|{amount_body}"
    runs: List[Tuple[int, int, str]] = []
    # Buscar todos los tokens cuantitativos
    for m in re.finditer(quant_token, s):
        tok = s[m.start():m.end()]
        if "%" in tok:
            continue
        is_decimal = bool(re.match(r"^\d+[.,]\d+$", tok) or re.match(r"^\d{1,3}(?:[.,]\d{3})+[.,]\d+$", tok))
        has_currency = bool(re.search(currency, tok))
        if is_decimal or has_currency:
            runs.append((m.start(), m.end(), tok))
    # Si hay más de un símbolo de divisa, dividir los tokens
    tokens = [tok for _, _, tok in runs]
    currency_count = sum(1 for t in tokens if re.search(currency, t))
    if currency_count > 1:
        # Dividir por cada símbolo de divisa encontrado
        split_tokens = re.findall(rf"{currency}\s*\d+(?:[.,]\d+)?", s)
        runs = []
        for match in split_tokens:
            start = s.find(match)
            end = start + len(match)
            runs.append((start, end, match))
    return runs

def is_decimal_number(s: str) -> bool:
        s = (s or "").strip()
        if not s or "%" in s:
            return False
        # casos: 2.98, 39,90, 1,000.00, 1.000,00
        patterns = [
            r"^\d+[.,]\d+$",                                # 2.98 / 39,90
            r"^\d{1,3}(?:[.,]\d{3})+[.,]\d+$",             # 1,000.00 / 1.000,00
        ]
        return any(re.match(p, s) for p in patterns)

def is_currency_amount(s: str) -> bool:
    s = (s or "").strip()
    if not s or "%" in s:
        return False

    currency_symbols = "$¢"
    currency = r"[$¢]"

    # Simple no-regex shortcut: if contains a currency and at least 1 digit after it
    for sym in currency_symbols:
        idx = s.find(sym)
        if idx != -1:
            after = s[idx+1 : ]
            if any(c.isdigit() for c in after):
                # No letras entre el símbolo y el número inmediato y no termina con símbolo y no es "00" después del símbolo y no digitos antes y símbolo al final
                maybe_amt = after.lstrip()
                # quick fail for 00 only amount
                possible_num = ""
                for ch in maybe_amt:
                    if ch.isdigit() or ch in ",.":
                        possible_num += ch
                    else:
                        break
                if possible_num == "00":
                    return False
                # Evitar simbolo al final
                if idx == len(s)-1:
                    return False
                # Evitar cantidades tipo "10.00$"
                if idx == 0 or (s[:idx].strip().isdigit() == False):
                    return True
                # En casos tipo "1,000$50", permitir
                before = s[:idx]
                if any(c.isdigit() for c in before):
                    return True
    # Si no coincide el shortcut, sigue el método previo por regex
    # El símbolo puede estar al inicio o en medio, pero NO al final
    pattern_start = rf"^{currency}\s*(\d{{1,3}}(?:[.,]\d{{3}})*|\d+)(?:[.,]\d+)?$"
    pattern_middle = (
        rf"^(\d{{1,3}}(?:[.,]\d{{3}})*|\d+)"
        rf"\s*{currency}\s*"
        rf"(\d{{1,3}}(?:[.,]\d{{3}})*|\d+)(?:[.,]\d+)?$"
    )
    pattern_end = rf"^(\d{{1,3}}(?:[.,]\d{{3}})*|\d+)(?:[.,]\d+)?\s*{currency}\s*$"
    multi_pattern = (
        rf"^(\s*{currency}\s*(\d{{1,3}}(?:[.,]\d{{3}})*|\d+)(?:[.,]\d+)?\s*){{2,}}$"
    )

    if re.match(pattern_end, s):
        return False

    cantidades = re.findall(rf"{currency}?\s*(\d+)(?:[.,]\d+)?\s*{currency}?", s)
    if any(c == "00" for c in cantidades):
        return False
    return (
        bool(re.match(pattern_start, s)) or
        bool(re.match(pattern_middle, s)) or
        bool(re.match(multi_pattern, s))
    )

def is_quantitative(token: str) -> bool:
    return is_currency_amount(token) or is_decimal_number(token)
            
def has_quantitative_pattern(s: str) -> bool:
    """
    Verifica si el texto contiene algún patrón cuantitativo (moneda o decimal).
    Busca patrones dentro del texto, incluso si hay caracteres basura.
    """
    if not s or not s.strip():
        return False
    
    # Verificar tokens separados por espacios
    tokens = [t for t in s.split() if t]
    if tokens:
        if any(is_quantitative(t) for t in tokens):
            return True
    
    # Si no hay espacios o no se encontró en tokens, buscar patrón dentro del string
    # Patrón para números decimales (con coma o punto como separador)
    decimal_pattern = r'\d{1,3}(?:[.,]\d{3})*[.,]\d{2,}'  # Ej: 1.275.00, 1,275.00
    
    # Patrón para moneda ($ € £ ¥ ¢ seguido de números)
    currency_pattern = r'[$€£¥¢]\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?'
    
    # Buscar cualquiera de estos patrones en el texto
    if re.search(decimal_pattern, s) or re.search(currency_pattern, s):
        return True
    
    # Último recurso: verificar el string completo limpio
    return is_quantitative(s)

def find_date(s: str) -> bool:
    try:
        if not s or not s.strip():
            return False

        moth_pattern = r'\b(ene(ro)?|feb(rero)?|mar(zo)?|abr(il)?|may(o)?|jun(io)?|jul(io)?|ago(sto)?|sep(tiembre)?|oct(ubre)?|nov(iembre)?|dic(iembre)?)\b'
        date_pattern = r'\b(\d{1,2})?[\/\-\. ]*(ene(ro)?|feb(rero)?|mar(zo)?|abr(il)?|may(o)?|jun(io)?|jul(io)?|ago(sto)?|sep(t(iembre)?)?|oct(ubre)?|nov(iembre)?|dic(iembre)?)[\/\-\. ]*(\d{2,4})?\b'

        # Busca primero el patrón corto
        if re.search(moth_pattern, s):
            # Si lo encuentra, busca el patrón largo

            if re.search(date_pattern, s):
                logger.info(f"Resultado de Date: {s}")
                return True

            else:
                return False

        # Si no encuentra el corto, busca el largo directamente
        if re.search(date_pattern, s):
            return True

        return False

    except Exception as e:
        logger.info(f"Error buscando Fecha: {e}", exc_info=True)
        return False
