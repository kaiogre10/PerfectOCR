import re
import logging
from typing import List, Tuple, Dict
from core.utils.text_encoder import validate_text

logger = logging.getLogger(__name__)

def is_acronym(text: str) -> bool:
    try:
        """
        Detecta siglas del tipo A.B.C. o P.U.C.D con punto final opcional
        y, a partir de ahora, admite un signo ':' ';' ',' opcional al final.
        """
        if not validate_text(text):
            return False
            
        pattern = r'^([A-Za-z]\.){2,}[A-Za-z]\.?(?:[:;,])?$'
        return re.search(pattern, text.strip()) is not None
        
    except Exception as e:
        logger.info(f"Error buscando siglas: {e}", exc_info=True)
    return False

def find_umd(s: str) -> bool:
    try:
        if not validate_text(s): #type: ignore
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
        if not validate_text(s): #type: ignore
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
        if not validate_text(s):
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
        
def contains_quantitative(s: str) -> bool:
    """
    Devuelve True si hay al menos un cuantitativo válido en cualquier parte del texto.
    """
    runs = find_quantitative_runs(s)
    return len(runs) > 0

def get_quantitative_patterns() -> Dict[str, str]:
    """
    Función interna que centraliza todos los patrones regex para reutilización.
    Ahora acepta la letra o/O como posible dígito para robustecer contra errores de OCR.
    """
    # Incluimos o y O como posibles dígitos
    digit = r"[0-9oO]"
    currency = r"[$¢]"
    amount_body = rf"(?:{digit}{{1,3}}(?:[.,]{digit}{{3}})*|{digit}+)(?:[.,]{digit}+)?"

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
    Ahora acepta la letra o/O como posible dígito para robustecer contra errores de OCR.
    """
    s = (s or "").strip()
    if not s or "%" in s:
        return False

    # Normaliza letras o/O a 0 para la validación final
    s_norm = s.replace("o", "0").replace("O", "0")

    patterns = get_quantitative_patterns()
    currency_symbols = "$¢"
    for sym in currency_symbols:
        idx = s_norm.find(sym)
        if idx != -1:
            after = s_norm[idx+1:]
            if any(c.isdigit() for c in after):
                maybe_amt = after.lstrip()
                possible_num = "".join(ch for ch in maybe_amt if ch.isdigit() or ch in ".,")
                if possible_num == "00":
                    return False
                if idx == len(s_norm) - 1: # Símbolo al final
                    return False
                if idx == 0 or not s_norm[:idx].strip().isdigit():
                    break # Es un candidato válido, proceder a regex

    if re.match(patterns["end"], s_norm):
        return False

    amounts = re.findall(r"\d+", s_norm)
    if any(c == "00" for c in amounts if len(amounts) > 1 or c != "00"):
        return False

    return bool(
        re.match(patterns["start"], s_norm) or
        re.match(patterns["middle"], s_norm) or
        re.match(patterns["multi"], s_norm) or
        re.match(r"^\d{1,3}(?:[.,]\d{3})*[.,]\d{2,}$", s_norm) # Decimal explícito
    )

def find_quantitative_runs(s: str) -> List[Tuple[int, int, str]]:
    """
    Encuentra TODAS las apariciones de entidades cuantitativas en un string,
    aceptando la letra o/O como posible dígito.
    """
    s = (s or "").strip()
    if not s:
        return []
    
    patterns = get_quantitative_patterns()
    runs: List[Tuple[int, int, str]] = []
    # Normaliza para la búsqueda
    s_norm = s.replace("o", "0").replace("O", "0")
    for m in re.finditer(patterns["token"], s_norm):
        tok = m.group(0)
        # Reutilizamos la lógica principal para validar cada token
        if find_quantitative(tok):
            runs.append((m.start(), m.end(), s[m.start():m.end()]))  # Devuelve el texto original

    # Lógica original para múltiples símbolos de divisa
    currency_count = sum(1 for _, _, tok in runs if re.search(patterns["currency"], tok))
    if currency_count > 1:
        split_runs: List[Tuple[int, int, str]] = []
        split_pattern = rf"{patterns['currency']}\s*{patterns['amount_body']}"
        for match in re.finditer(split_pattern, s_norm):
            split_runs.append((match.start(), match.end(), s[match.start():match.end()]))
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
