# core/utils/text_validator.py
from typing import Set, Dict
import unicodedata
import re

def validate_text(text: str) -> bool:
    if text.isspace():
        return False

    elif len(text) > 0:
        return True

    else:
        return False
    
def get_char_num() -> Set[str]:
     return {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", ",", "$"}
    
def get_special_chars() -> Set[str]:
    return {
        ")", "(", "]", "[", "{", "}", "|", "*", "^",
        "-", "_", "+", "=", "<", ">", ";", ":", "@",
        "'", "!", "¡", "?", "¿", "'", "\\", "''",
    }
def numeric_corrections() -> Dict[str, str]:
    return {
        "Q": "0",
        "q": "9",
        "O": "0",
        "o": "0",
        "I": "1",
        "i": "1",
        "|": "1",
        "!": "1",
        "¡": "1",
        "l": "1",
        #"S": "$",
        "s": "5",
        "G": "6",
        "g": "9",
        "B": "8",
        "Z": "2",
        "z": "2",
        "j": "9"
        }

def descritive_corrections() -> Dict[str, str]:
    return {
        "$": "S", 
        "è": "é", 
        "ý": "y", 
        "\\": "/",
    }

def not_valid_chars() -> Set[str]:
    return {
        "~",
        "©",
        "®",
        "™",
        "`",
        "¬",
        "¨",
        "÷",
        "°",
        "=",
    }

def valid_punt_chars() -> Set[str]:
    not_valid_punt_chars: Set[str] = {
        ".", "*", "^", "°", ",",
        "-", "_", ";", ":",
        "'",  "´", "''", "¨"
    }
    return not_valid_punt_chars.union(not_valid_chars())

def punc_chars() -> Set[str]:
    return {".", ";", ":", "!", "?"}

def get_alone_chars() -> Set[str]:
    return {"a", "e", "y", "o", "u", "&"}

def validate_alone_chars(text: str) -> bool:
    text = text.strip().lower()
    valid_chars = get_alone_chars()
    if len(text) > 1:
        return True

    elif text.isdecimal():
        return True
    
    elif text in valid_chars:
        return True

    else:
        return False
        
def norm_text(text: str) -> str:
    if not validate_text(text):
        return ""

    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8').lower()

def is_upper(text: str) -> bool:
    uppers: int = 0
    text = text.strip()
    for char in text:
        if char.islower():
            continue
        uppers += 1
    upper_mean = uppers/len(text)
    # print(f"{upper_mean}")
    if upper_mean > 0.666:
        return True
    else:
        return False

def estandarice_uppers_lowers(text_base: str, clean_text: str) -> str:
    if text_base.isupper() or is_upper(text_base):
        return clean_text.upper()
    elif text_base.islower():
        return clean_text.lower()
    elif text_base.istitle():
        return clean_text.title()
    else:
        return clean_text

def space_removal(text: str) -> str:
    """
    Limpia espacios múltiples y espacios iniciales/finales de un texto.
    Reemplaza múltiples espacios consecutivos por un solo espacio y elimina espacios al inicio y final.
    """
    if not text:
        return ""
    # Reemplazar múltiples espacios consecutivos por un solo espacio
    clean_text = re.sub(r"\s+", " ", text).strip()
    # Eliminar espacios iniciales y finales
    if not clean_text or not validate_text(clean_text):
        return ""
    else:
        return clean_text

