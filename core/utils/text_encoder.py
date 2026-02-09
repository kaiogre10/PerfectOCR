# core/utils/text_encoder.py
from core.utils.text_validator import validate_text, get_char_num
from core.utils.data_utils import DENSITY_ENCODER, CHAR_FRECUENCY, INV_FRECUENCY_ENCODER
from typing import List, Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

density = DENSITY_ENCODER
frecuency = CHAR_FRECUENCY
inverse = INV_FRECUENCY_ENCODER

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
    if not validate_text(text):
        return " "
    else:
        minus_text = text.lower()
        return ''.join(minus_text.split())
    
def text_encode(text: str, encoding_type: List[str]) -> np.ndarray[Any, np.dtype[np.float32]]:
    if "all" in encoding_type and len(encoding_type) == 1:
        encoding_type = ["density", "inverse", "frequency", "morphological"]

    encoders: List[List[float]]= []
    for enc_type in encoding_type:

        if enc_type == "density":
            dense = encode_text(text,density)
            encoders.append(dense)
        if enc_type == "inverse":
            inv = encode_text(text, inverse)
            encoders.append(inv)
        if enc_type == "frequency":
            frec = encode_text(text, frecuency)
            encoders.append(frec)
        if enc_type == "morphological":
           morph = get_morphological_encode(text)
           encoders.append(morph)
    
    return np.array(encoders, np.float32)
        

        
