# PerfectOCR/core/workers/ocr/text_corrector.py
import logging
import dataclasses
from typing import Dict, Any, List
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons
from core.factory.abstract_worker import OCRAbstractWorker
from core.utils.text_utils import  validate_text, find_umd, get_brands
from core.utils.data_utils import CUANT_CHAR, NUMERIC_CORRECTIONS, DESCRIPTIVE_CORRECTIONS, UMD_CORRECTIONS, NOT_VALID_CHARS

cuant_char = CUANT_CHAR
numeric_corrections = NUMERIC_CORRECTIONS
des_corrects = DESCRIPTIVE_CORRECTIONS
umd_corrects = UMD_CORRECTIONS
not_valid = NOT_VALID_CHARS

logger = logging.getLogger(__name__)

class TextCorrector(OCRAbstractWorker):
    """
    - Corrector textual quirúrgico que realiza reemplazos especializados de caracteres según el tipo semántico de cada polígono:
    - Según la clasificación semántica aplica correcciones específicas.
    - Solo hace reemplazos de caracteres, no corrección ortográfica.
    - Es recursivo: itera sobre todos los polígonos aplicando correcciones especializadas.
    """
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
            
    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        logger.debug(f"Inicia text_corrector")
        if not manager.workflow or not manager.workflow.polygons:
            logger.warning("TextCorrector: No hay polígonos para procesar.")
            return False
            
        polygons_in: Dict[str, Polygons] = manager.workflow.polygons

        logger.debug(f"Cantidad de polígonos recibidos:{len(polygons_in)}")
        corrected_polygons: Dict[str, Polygons] = {}
        correced_count = 0

        for poly_id, polygon in polygons_in.items():
            original_text = polygon.ocr_text or ""
            original_text = original_text.strip()
            kf = polygon.key_field
            sc =  polygon.semantic_clasification
            
            if kf or kf is not None:
                # logger.info(f"'{poly_id}' con KEYFIELD ya no se CORRIJE: '{original_text}'")
                updated_polygon = dataclasses.replace(polygon)
                corrected_polygons[poly_id] = updated_polygon
                continue

            if any(c in (0, 5) for c in sc):
                # logger.info(f"'{poly_id}' NUMERICO ya no se CORRIJE: '{original_text}'")
                updated_polygon = dataclasses.replace(polygon)
                corrected_polygons[poly_id] = updated_polygon
                continue
                
            # Si el texto está vacío, no hay nada que corregir
            if not original_text or not validate_text(original_text):
                # logger.info(f"Sin Texto válido: {poly_id}: '{original_text}'")
                continue
            
            # Aplicar corrección según tipo semántico
            corrected_text = self._apply_corrections(original_text, sc)
            # Si hubo cambios, actualizar el polígono
            if not corrected_text or not validate_text(corrected_text):
                # logger.info(f"Sin Texto válido: {poly_id}: '{original_text}'")
                continue

            if corrected_text != original_text:
                correced_count +=1
                logger.debug(
                    f"Corrección para '{poly_id}':"
                    f"Original: '{original_text}' → Corregido: '{corrected_text}' SC: {sc}"
                )
                
            updated_polygon = dataclasses.replace(polygon, ocr_text=corrected_text)
            corrected_polygons[poly_id] = updated_polygon
                
        manager.workflow.polygons = corrected_polygons
        return True

    def _apply_corrections(self, text: str, semantic_clasification: List[int]) -> str:
        text = text.strip()
        tokens = text.split(' ')
        total_tokens = len(tokens)

        if not tokens or not total_tokens:
            return ""
        
        if total_tokens != len(semantic_clasification):
            logger.warning(f"Desalineación en '{text}': {total_tokens} tokens vs {len(semantic_clasification)} clasificaciones. SC: {semantic_clasification}")
            return text
        
        if total_tokens == 1:
            return self._correct_token(text, semantic_clasification[0])
        
        corrected_tokens: List[str] = []

        for i, token in enumerate(tokens):
            token_sc = semantic_clasification[i]
            corrected_token = self._correct_token(token, token_sc)
            if not any(c.isalnum() for c in corrected_token):
                continue

            elif bool(i == 0 or (i + 1) == total_tokens):
                if validate_text(corrected_token):
                    corrected_tokens.append(corrected_token)
                    continue
                continue
            else:
                corrected_tokens.append(corrected_token)

        return ' '.join(corrected_tokens)

    def _correct_token(self, token: str, semantic_clasification: int) -> str:
        if not any(c.isalnum() for c in token):
            return ""

        elif len(token) == 1:
            if semantic_clasification == 1 and "0" in token:
                return token.replace("0", "O")
            return token
            
        if token.isalpha():
            return token

        if token.isdecimal():
            return token
        
        if semantic_clasification in (4, 5):
            return self._correct_cuants(token)
            
        elif semantic_clasification in (1, 2):
            if token.endswith("m1"):
                token = token.replace("1", "l")
                
            if get_brands(token):
                token = token.replace("1", "I")
                
            if semantic_clasification == 1:
                return token.replace("0", "O")
                
            return find_umd(token)
            
        # elif semantic_clasification == 3:
            # return token
        else:
            return token
    
    def _correct_cuants(self, token: str) -> str:
        corrected_chars = [numeric_corrections.get(ch, ch) for ch in token]
        return ''.join(corrected_chars)