# core/utils/text_encoder.py
from core.utils.text_validator import validate_text, get_char_num
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

def encode_text(text: str, encoder: Dict[str, float]) -> List[float]:
    try:
        if not validate_text(text): 
            return []

        compact_text = text_compacter(text)
        encoded_poly = [encoder.get(char, 0) for char in compact_text]

        return encoded_poly

    except Exception as e:
        logger.warning(f"Error codificando polígonos: {e}", exc_info=True)
    return []
                
def get_morphological_encode(text: str) -> List[float]:
    try:
        result: List[float] = []
        if not validate_text(text):
            return []
        
        char_num = get_char_num()
        if not char_num:
            return []

        compact_text = text_compacter(text)
        for ch in compact_text:
            if ch in char_num:
                result.append(1.0)
            elif ch.isalpha():
                result.append(-1.0)
            else:
                result.append(0.0)
        return result

    except Exception as e:
        logger.warning(f"Error codificando polígonos: {e}", exc_info=True)
    return []

def text_compacter(text: str) -> str:
    minus_text = text.lower()
    return ''.join(minus_text.split())