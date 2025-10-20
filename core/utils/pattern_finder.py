import re
import logging
from typing import List, Tuple, Dict

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

def get_quantitative_patterns() -> Dict[str, str]:
    """
    Función interna que centraliza todos los patrones regex para reutilización.
    """
    currency = r"[$¢]"
    amount_body = r"(?:\d{1,3}(?:[.,]\d{3})*|\d+)(?:[.,]\d+)?"

    return {
        "currency": currency,
        "amount_body": amount_body,
        "token": rf"{currency}\s*{amount_body}|{amount_body}\s*{currency}|{amount_body}",
        "start": rf"^{currency}\s*{amount_body}$",
        "middle": rf"^{amount_body}\s*{currency}\s*{amount_body}$",
        "end": rf"^{amount_body}\s*{currency}$",
        "multi": rf"^(?:\s*{currency}\s*{amount_body}\s*){{2,}}$"
    }

def find_quantitative(s: str) -> bool:
    """
    Determina si el string COMPLETO es una única entidad cuantitativa.
    Consolida toda la lógica de validación de find_quantitative y find_numeric.
    """
    s = (s or "").strip()
    if not s or "%" in s:
        return False

    patterns = get_quantitative_patterns()
    currency_symbols = "$¢"
    for sym in currency_symbols:
        idx = s.find(sym)
        if idx != -1:
            after = s[idx+1:]
            if any(c.isdigit() for c in after):
                maybe_amt = after.lstrip()
                possible_num = "".join(ch for ch in maybe_amt if ch.isdigit() or ch in ".,")
                if possible_num == "00":
                    return False
                if idx == len(s) - 1: # Símbolo al final
                    return False
                if idx == 0 or not s[:idx].strip().isdigit():
                    break # Es un candidato válido, proceder a regex

    if re.match(patterns["end"], s):
        return False

    amounts = re.findall(r"\d+", s)
    if any(c == "00" for c in amounts if len(amounts) > 1 or c != "00"):
        return False

    return bool(
        re.match(patterns["start"], s) or
        re.match(patterns["middle"], s) or
        re.match(patterns["multi"], s) or
        re.match(r"^\d{1,3}(?:[.,]\d{3})*[.,]\d{2,}$", s) # Decimal explícito
    )

def find_quantitative_runs(s: str) -> List[Tuple[int, int, str]]:
    """
    Encuentra TODAS las apariciones de entidades cuantitativas en un string,
    reutilizando la lógica de quantitative_runs original.
    """
    s = (s or "").strip()
    if not s:
        return []
    
    patterns = get_quantitative_patterns()
    runs: List[Tuple[int, int, str]] = []
    for m in re.finditer(patterns["token"], s):
        tok = m.group(0)
        # Reutilizamos la lógica principal para validar cada token
        if find_quantitative(tok):
            runs.append((m.start(), m.end(), tok))
            
    # Lógica original para múltiples símbolos de divisa
    currency_count = sum(1 for _, _, tok in runs if re.search(patterns["currency"], tok))
    if currency_count > 1:
        # Si hay múltiples monedas, se divide por cada una
        split_runs: List[Tuple[int, int, str]] = []
        split_pattern = rf"{patterns['currency']}\s*{patterns['amount_body']}"
        for match in re.finditer(split_pattern, s):
            split_runs.append((match.start(), match.end(), match.group(0)))
        return split_runs

    return runs

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
