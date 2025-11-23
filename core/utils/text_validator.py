# core/utils/text_validator.py
from typing import Set, Dict

def validate_text(text: str) -> bool:
    if len(text) > 0:
        return True
    else:
        return False
    
def get_char_num() -> Set[str]:
    char_num: Set[str] = set(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", ",", "$"])
    return char_num
    
def get_special_chars() -> Set[str]:
    return {
        ")", "(", "]", "[", "{", "}", "|", "*", "^",
        "-", "_", "+", "=", "<", ">", ";", ":", "@",
        "'", "!", "¡", "?", "¿", "'", "\\", "''",
    }
def numeric_corrections() -> Dict[str, str]:
    return {
        "O": "0",
        "o": "0",
        "I": "1",
        "i": "1",
        "|": "1",
        "l": "1",
        "S": "$",
        # "s": "5",
        "G": "6",
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
        "0": "O"
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
        "°"
    }

def valid_punt_chars() -> Set[str]:
    not_valid_punt_chars: Set[str] = set({
        ".", "*", "^", "°", ",",
        "-", "_", "=",  ";", ":",
        "'",  "´", "''", "¨"
    })
    return not_valid_punt_chars.union(not_valid_chars())

def punc_chars() -> Set[str]:
    return {".", ";", ":", "!", "?"}

def get_alone_chars() -> Set[str]:
    return {"a", "e", "y", "o", "u", "&"}

def validate_alone_chars(text: str) -> bool:
    if len(text) > 1:
        return True
    
    minus_text = text.lower()
    
    nums = get_char_num()
    if minus_text in nums:
        return True
    
    valid_chars = get_alone_chars()
    if minus_text in valid_chars:
        return True

    spec_char = get_special_chars()
    not_valid_chr = not_valid_chars()
    if minus_text in spec_char or minus_text in not_valid_chr:
        return False
    
    else:
        return False
