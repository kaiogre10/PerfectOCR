# core/utils/text_validator.py
from fuzzywuzzy import utils #type: ignore
from typing import List, Dict

def validate_text(text: str) -> bool:
    if utils.validate_string(text): #type: ignore
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