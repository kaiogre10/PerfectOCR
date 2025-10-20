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
