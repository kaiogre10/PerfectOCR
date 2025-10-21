from fuzzywuzzy import utils #type:ignore
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

def encode_text(text: str, encoder: Dict[str, float]) -> List[float]:
        try:
            if not utils.validate_string(text): #type:ignore
                return []

            if text:

                minus_text = text.lower()
                compact_text = ''.join(minus_text.split())
                encoded_poly = [encoder.get(char, 0) for char in compact_text]

                return encoded_poly

        except Exception as e:
            logger.warning(f"Error codificando polígonos: {e}", exc_info=True)
        return []
        
def get_morphological_map(text: str) -> List[float]:
    char_num: List[str] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", ",", "$"]
    try:
        result: List[float] = []
        if not utils.validate_string(text):  # type:ignore
            return []

        compact_text = ''.join(text.split())
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