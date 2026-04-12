# PerfectOCR/core/workers/ocr/text_corrector.py
import logging
import dataclasses
from typing import Dict, Any, List
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons
from core.factory.abstract_worker import OCRAbstractWorker
from core.utils.text_utils import  validate_text, clean_cuant, validate_quant_chars, find_umd
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
                updated_polygon = dataclasses.replace(polygon, ocr_text=corrected_text)
                corrected_polygons[poly_id] = updated_polygon
                logger.debug(
                    f"Corrección para '{poly_id}':"
                    f"Original: '{original_text}' → Corregido: '{corrected_text}' SC: {sc}"
                )
            else:
                corrected_polygons[poly_id] = polygon
                
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
            return token

        if token.isalpha():
            return token

        if token.isdecimal():
            return token
        
        elif semantic_clasification == 4:
            return token
            dec_token = clean_cuant(token)
            cor = self._correct_cuants(dec_token)
            # logger.info(f"Correct cuants: '{token}' -> '{dec_token}' -> '{cor}")
            return cor

        elif semantic_clasification == 2:
            return find_umd(token)

        else:
            return token
    
            # 1. Primero corregir chars del token completo

        #     # 2. Luego separar por runs cuantitativos
        #     parts = self.split_by_quantitative_runs(corrected_token)

        # for part, is_quant in parts:
        #     # 3. Aplicar numeric_separator solo a partes cuantitativas
        #     final_part = numeric_separator(part) if is_quant else part
        #     if final_part and validate_unique_chars(final_part):
        #         corrected_tokens.append(final_part)
        # else:
        # corrected_token = self._correct_token(token, token_sc)
        # if corrected_token and validate_unique_chars(corrected_token):
        #     corrected_tokens.append(corrected_token)    
    
    def _correct_cuants(self, token: str) -> str:
        corrected_chars = [numeric_corrections.get(ch, ch) for ch in token]
        return ''.join(corrected_chars)    
    
        # """Aplica correcciones a un único token basado en su clasificación semántica."""

        # # corrections_map = self._get_corrections_map(semantic_clasification)
        # if not corrections_map:
        #     return token
        
        # # token = token.strip(self.not_valid_chars)
        # corrected_chars = list(token)

        # for i, char in enumerate(token):
        #     if char not in corrections_map:
        #         continue
            
        #     if char.isalnum() and not self._is_isolated(token, i):
        #         continue
        #     if char == "0" and self.termination_correct(token, semantic_clasification):
        #         replacement = "o"
        #     elif char == "S" and self._should_use_five_instead_of_dollar(token, i, semantic_clasification):
        #         replacement = "5"
        #     else:
        #         replacement = corrections_map[char]
        #     corrected_chars[i] = replacement

        # # numeric_separator ya NO se aplica aquí para sc==2
        # return ''.join(corrected_chars)

    # def _is_solated2(self, text: str):
    #     return (sum(1 for char in text if char.isdecimal()) == 1 or sum(1 for char in text if char.isalpha()) == 1) if text.isalnum() else False


    # def _is_isolated(self, text: str, index: int) -> bool:
    #     """
    #     Verifica si un carácter está AISLADO (sin vecinos del mismo tipo).
    #     Ignora espacios al buscar vecinos.
    #     """
    #     if index < 0 or index > len(text):
    #         return False

    #     current_char = text[index]
    #     current_is_digit = current_char in cuant_char
    #     current_is_alpha = current_char.isalpha()
        
    #     # Si no es letra ni número, no aplicar corrección
    #     if not current_is_digit and not current_is_alpha:
    #         return False
        
    #     # Buscar vecino izquierdo (ignorando espacios)
    #     left_neighbor = None
    #     for i in range(index - 1, -1, -1):
    #         if text[i] != ' ':
    #             left_neighbor = text[i]
    #             break
        
    #     # Buscar vecino derecho (ignorando espacios)
    #     right_neighbor = None
    #     for i in range(index + 1, len(text)):
    #         if text[i] != ' ':
    #             right_neighbor = text[i]
    #             break
        
    #     # Verificar si NINGÚN vecino es del mismo tipo (está aislado)
    #     has_left_match = False
    #     has_right_match = False
        
    #     if left_neighbor:
    #         if current_is_digit and left_neighbor in cuant_char:
    #             has_left_match = True
    #         elif current_is_alpha and left_neighbor.isalpha():
    #             has_left_match = True
        
    #     if right_neighbor:
    #         if current_is_digit and right_neighbor in cuant_char:
    #             has_right_match = True
    #         elif current_is_alpha and right_neighbor.isalpha():
    #             has_right_match = True
        
    #     # Está aislado si NO tiene ningún vecino del mismo tipo
    #     return not (has_left_match or has_right_match)
        
    # def _get_corrections_map(self, semantic_clasification:  int) -> Dict[str, str]:
    #     """Devuelve el mapa de correcciones para un tipo semántico dado."""
    #     if not NUMERIC_CORRECTIONS:
    #         logger.error("Sin mapa de correcciones")
    #         return {}

    #     if semantic_clasification == 1:
    #         return NUMERIC_CORRECTIONS
    #     elif semantic_clasification == 2:
    #         return NUMERIC_CORRECTIONS
    #     elif semantic_clasification == 0:
    #         return DESCRIPTIVE_CORRECTIONS
    #     elif semantic_clasification == -2:
    #         return UMD_CORRECTIONS
    #     else:
    #         return {}
        
    # def _should_use_five_instead_of_dollar(self, text: str, index: int, semantic_clasification:  int) -> bool:
    #     if semantic_clasification not in (1, 2):
    #         return False

    #     if index < 0 or index >= len(text):
    #         return False
            
    #     if text[index] != 'S':
    #         return False

    #     # Delimitar token por espacios
    #     l = index
    #     while l > 0 and text[l-1] != ' ':
    #         l -= 1
    #     r = index
    #     while r + 1 < len(text) and text[r+1] != ' ':
    #         r += 1
    #     token = text[l:r+1]

    #     # Debe ser cuantitativo real (contener dígitos)
    #     if not any(ch in cuant_char for ch in token):
    #         return False

    #     # Si 'S' está al inicio del token: NO forzar '5' (permitir '$')
    #     if index == l:
    #         return False

    #     # Contexto numérico local (vecinos)
    #     left = text[index-1] if index-1 >= l else ' '
    #     right = text[index+1] if index+1 <= r else ' '
    #     left_numish = left.isdecimal() or left in '.,'
    #     right_numish = right.isdecimal() or right in '.,'

    #     # Si hay '$' antes en el token o está en contexto numérico → usar '5'
    #     has_currency_before = '$' in token[:index - l]
    #     return has_currency_before or left_numish or right_numish

    # # def termination_correct(self, text: str, semantic_clasification: int) -> bool:
    # #     if semantic_clasification != 0:
    # #         return False

    # #     if termination_detect(text):
    # #         return True
    # #     else:
    # #         return False
