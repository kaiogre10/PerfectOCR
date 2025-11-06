# core/utils/text_validator.py
from typing import List, Dict

def validate_text(text: str) -> bool:
    if len(text) > 0:
        return True
    else:
        return False
    
def get_char_num() -> List[str]:
    char_num: List[str] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", ",", "$"]
    return char_num

def not_valid_chars() -> List[str]:
    return [
        "~",
        "©",
        "®",
        "™",
        "`",
        "¬",
        "¨",
        "÷"
        ]
    
def special_chars() -> List[str]:
    return [
        ")", "(", "]", "[", "{", "}", "|", "*", "^",
        "-", "_", "+", "=", "<", ">", ";", ":", "@",
        "'", "!", "¡", "?", "¿", "'", "\\", "''",
        ]
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

def punc_chars() -> List[str]:
    return [".", ";", ":", "!", "?"]

def get_alone_chars() -> List[str]:
    return ["a", "e", "y", "o", "u", "&"]

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

    spec_char = special_chars()
    if minus_text in spec_char:
        return False
    
    else:
        return False
