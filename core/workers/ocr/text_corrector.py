# PerfectOCR/core/workers/ocr/text_corrector.py
import logging
import dataclasses
from typing import Dict, Any, List
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons
from core.factory.abstract_worker import OCRAbstractWorker
from core.utils.text_encoder import get_char_num, validate_text
from core.utils.text_validator import numeric_corrections, descritive_corrections
from core.utils.pattern_finder import termination_detect

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
        self.worker_config = config.get("text_corrector", {})
        self.conf_threshold = (self.worker_config.get("confidence_threshold") * 100.0)
        self.numeric_corrections: Dict[str, str] = numeric_corrections()
        self.descriptive_corrections: Dict[str, str] = descritive_corrections()
        self.char_num: List[str] = get_char_num()
            
    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:

        if not manager.workflow or not manager.workflow.polygons:
            logger.warning("TextCorrector: No hay polígonos para procesar.")
            return False
            
        polygons_in: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}

        corrected_polygons: Dict[str, Polygons] = polygons_in
        correction_stats: Dict[str, int] = {
            "numeric": 0,
            "quantitative": 0,
            "code": 0,
            "umd": 0,
            "descriptive": 0,
            "total_corrections": 0,
            "skipped_high_confidence": 0
        }
        
        sorted_poly_ids = sorted(polygons_in.keys())
        logger.info(f"Cantidad de polígonos recibidos:{len(sorted_poly_ids)}")
        # Procesar cada polígono recursivamente
        for poly_id in sorted_poly_ids:
            polygon = polygons_in[poly_id]
            original_text = polygon.ocr_text or ""
            confidence = polygon.ocr_confidence or 0.0
            
            # Si el texto está vacío, no hay nada que corregir
            if not validate_text(original_text):
                logger.debug(f"Sin texto: {poly_id}: '{original_text}'")
                continue

            # token_corr = correct_termination(token)
            
            # Filtro de confianza
            if confidence > self.conf_threshold:
                corrected_polygons[poly_id] = polygon
                correction_stats["skipped_high_confidence"] += 1
                continue
            
            # Aplicar corrección según tipo semántico
            corrected_text = self._apply_corrections(text=original_text, semantic_clasification=polygon.semantic_clasification, polygon_id=poly_id)
            # Si hubo cambios, actualizar el polígono
            if corrected_text != original_text:
                updated_polygon = dataclasses.replace(polygon, ocr_text=corrected_text, was_refined=True)
                corrected_polygons[poly_id] = updated_polygon
                sc = polygon.semantic_clasification
                semantic_type = "numeric" if sc == 1 else "quantitative" if sc == 2 else "descriptive" if sc == 0 else "umd" if sc == -2 else "code"
                correction_stats[semantic_type] += 1
                correction_stats["total_corrections"] += 1
                
                logger.info(
                    f"Corrección {poly_id}: "
                    f"Tipo: '{semantic_type}' | "
                    f"Confianza: '{confidence}' | "
                    f"Original: '{original_text}' → Corregido: '{corrected_text}'"
                )
            else:
                corrected_polygons[poly_id] = polygon
                
        manager.workflow.polygons = corrected_polygons
        
        logger.debug(f"Total correcciones: {correction_stats['total_corrections']}")
        logger.debug(
            f"Corrección textual - "
            f"Alta confianza omitidos: {correction_stats['skipped_high_confidence']} | "
            f"Numeric: {correction_stats['numeric']} | "
            f"Quantitative: {correction_stats['quantitative']} | "
            f"Code: {correction_stats['code']} | "
            f"UMD: {correction_stats['umd']} | "
            f"Descriptive: {correction_stats['descriptive']}"
        )

        return True

    def _apply_corrections(self, text: str, semantic_clasification:  List[int] | int, polygon_id: str) -> str:
        if not validate_text(text):
            return text

        tokens = text.split(' ')
        
        # Si la clasificación es una lista, debe corresponder con los tokens
        is_list_classification = isinstance(semantic_clasification, list)
        logger.debug(f"Procesando {polygon_id}: '{text}' | Clasificación: {semantic_clasification}")
        if is_list_classification and len(semantic_clasification) != len(tokens):
            logger.warning(f"Discrepancia en {polygon_id}: {len(tokens)} tokens vs {len(semantic_clasification)} clasificaciones. No se corrige.")
            return text

        corrected_tokens: List[str] = []
        for i, token in enumerate(tokens):
            # Obtener la clasificación para el token actual
            token_sc = semantic_clasification[i] if is_list_classification else semantic_clasification

            # Lógica de corrección principal (movida a un método auxiliar)
            corrected_token = self._correct_token(token, token_sc, polygon_id)
            corrected_tokens.append(corrected_token)
            
        return ' '.join(corrected_tokens)
    
    def _correct_token(self, token: str, semantic_clasification: int, polygon_id: str) -> str:
        """Aplica correcciones a un único token basado en su clasificación semántica."""
        if semantic_clasification in [-1, -2]:
            semantic_type = "umd" if semantic_clasification == -2 else "code"
            logger.debug(f"Omitiendo corrección de token en {polygon_id}: '{semantic_type}', '{token}'")
            return token
            
        # Seleccionar el diccionario de correcciones apropiado
        corrections_map = self._get_corrections_map(semantic_clasification)
        
        if not corrections_map:
            return token
            
        # Aplicar reemplazos quirúrgicos solo si el carácter está AISLADO
        corrected_chars = list(token)
        
        for i, char in enumerate(token):
            if char not in corrections_map:
                continue

            # Mantener tu chequeo de aislamiento
            if not self._is_isolated(token, i):
                logger.debug(f"{polygon_id}: Carácter '{char}' en '{token}' no está aislado, se omite corrección.")
                continue

            if char == "0" and self.termination_correct(token, semantic_clasification):
                replacement = 'o'
                logger.info(f"{polygon_id}: Corrección de terminación")

            if char == 'S' and self._should_use_five_instead_of_dollar(token, i, semantic_clasification):
                replacement = '5'
                logger.info(f"{polygon_id}: Regla especial S->5 en '{token}'")
            else:
                replacement = corrections_map[char]

            logger.info(f"{polygon_id}: Corrigiendo: '{char}' => '{replacement}'en texto original: '{token}'")
            corrected_chars[i] = replacement
            
        return ''.join(corrected_chars)

    def _is_isolated(self, text: str, index: int) -> bool:
        """
        Verifica si un carácter está AISLADO (sin vecinos del mismo tipo).
        Ignora espacios al buscar vecinos.
        """
        if index < 0 or index >= len(text):
            return False
            
        current_char = text[index]
        current_is_digit = current_char in self.char_num
        current_is_alpha = current_char.isalpha()
        
        # Si no es letra ni número, no aplicar corrección
        if not current_is_digit and not current_is_alpha:
            return False
        
        # Buscar vecino izquierdo (ignorando espacios)
        left_neighbor = None
        for i in range(index - 1, -1, -1):
            if text[i] != ' ':
                left_neighbor = text[i]
                break
        
        # Buscar vecino derecho (ignorando espacios)
        right_neighbor = None
        for i in range(index + 1, len(text)):
            if text[i] != ' ':
                right_neighbor = text[i]
                break
        
        # Verificar si NINGÚN vecino es del mismo tipo (está aislado)
        has_left_match = False
        has_right_match = False
        
        if left_neighbor:
            if current_is_digit and left_neighbor in self.char_num:
                has_left_match = True
            elif current_is_alpha and left_neighbor.isalpha():
                has_left_match = True
        
        if right_neighbor:
            if current_is_digit and right_neighbor in self.char_num:
                has_right_match = True
            elif current_is_alpha and right_neighbor.isalpha():
                has_right_match = True
        
        # Está aislado si NO tiene ningún vecino del mismo tipo
        return not (has_left_match or has_right_match)
        
    def _get_corrections_map(self, semantic_clasification:  int) -> Dict[str, str]:
        """Devuelve el mapa de correcciones para un tipo semántico dado."""
        if not self.numeric_corrections:
            logger.error("Sin mapa de correcciones")
            return {}

        if semantic_clasification == 1:
            return self.numeric_corrections
        elif semantic_clasification == 2:
            return self.numeric_corrections
        elif semantic_clasification == 0:
            return self.descriptive_corrections
        else:
            return {}
        
    def _should_use_five_instead_of_dollar(self, text: str, index: int, semantic_clasification:  int) -> bool:
        if semantic_clasification not in (1, 2):
            return False

        if index < 0 or index >= len(text):
            return False
            
        if text[index] != 'S':
            return False

        # Delimitar token por espacios
        l = index
        while l > 0 and text[l-1] != ' ':
            l -= 1
        r = index
        while r + 1 < len(text) and text[r+1] != ' ':
            r += 1
        token = text[l:r+1]

        # Debe ser cuantitativo real (contener dígitos)
        if not any(ch.isdigit() for ch in token):
            return False

        # Si 'S' está al inicio del token: NO forzar '5' (permitir '$')
        if index == l:
            return False

        # Contexto numérico local (vecinos)
        left = text[index-1] if index-1 >= l else ' '
        right = text[index+1] if index+1 <= r else ' '
        left_numish = left.isdigit() or left in '.,'
        right_numish = right.isdigit() or right in '.,'

        # Si hay '$' antes en el token o está en contexto numérico → usar '5'
        has_currency_before = '$' in token[:index - l]
        return (has_currency_before or left_numish or right_numish)

    def termination_correct(self, text: str, semantic_clasification: int) -> bool:
        if semantic_clasification != 0:
            return False

        if termination_detect(text):
            return True
        else:
            return False
