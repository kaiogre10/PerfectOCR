# PerfectOCR/core/workers/ocr/text_corrector.py
import logging
import time
from typing import Dict, Any, List
from domain.data_formatter import DataFormatter
from domain.abstract_worker import OCRAbstractWorker
from utils.text_utils import find_umd, fast_classfier, correct_subfix
from core.assets.assets import NUMERIC_CORRECTIONS, UMD_CORRECTIONS
from core.assets.patterns import bad_title
from utils.compiled_utils import validate_text
from string import ascii_lowercase, ascii_uppercase

_bad_title = bad_title
numeric_corrections = NUMERIC_CORRECTIONS
umd_corrects = UMD_CORRECTIONS

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
        self.output_log = config.get("text_correct")
        self.del_output_log = config.get("text_del")
            
    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        t0 = time.perf_counter()
        if not manager.workflow or not manager.workflow.polygons:
            logger.warning("TextCorrector: No hay polígonos para procesar.")
            return False
            
        polygons_in = manager.workflow.polygons if manager.workflow else {}
        if not polygons_in:
            logger.error("No hay polygons para procesar", exc_info=True)
            return False

        final_polygons: Dict[str, Dict[str, Any]] = {}
        correced_count = 0

        for poly_id, polygon in polygons_in.items():
            original_text = polygon.ocr_text or ""
            kf = polygon.key_field
            sc = polygon.semantic_clasification
            
            if 0 in sc and kf is not None:
                # logger.info(f"'{poly_id}' con KEYFIELD ya no se CORRIJE: '{original_text}'")
                final_polygons[poly_id] = {"text": original_text, "sc": sc, "cuant_chars": polygon.cuant_chars}
                continue

            if len(original_text.split(" ")) != len(sc):
                logger.critical(f"ERROR CRÍTICO DISPARIDAD EN {poly_id} ENTRE SC Y TEXTO: '{original_text}' -> {sc} ABORTANDO PROCESO")
                return False
                
            # Si el texto está vacío, no hay nada que corregir
            elif not original_text or not validate_text(original_text):
                if self.del_output_log:
                    logger.info(f"Sin Texto válido: {poly_id}: '{original_text}'")
                correced_count +=1
                continue
            
            # Aplicar corrección según tipo semántico
            corrected_text = self._apply_corrections(original_text, sc)
            if not corrected_text or not validate_text(corrected_text):
                if self.del_output_log:
                    logger.info(f"Sin Texto válido: {poly_id}: '{corrected_text}'")
                correced_count +=1
                continue

            else:
                if corrected_text != original_text:
                    s_class, t_cuan = fast_classfier(corrected_text)
                    if self.output_log:
                        logger.info(f"Corrección de '{poly_id}' | Original: '{set(original_text.split(" ")).difference(set(corrected_text.split(" ")))}' → '{corrected_text}' | SC original: {sc} -> {s_class}")
                    final_polygons[poly_id] = {"text": corrected_text, "sc": s_class, "cuant_chars": t_cuan}

                final_polygons[poly_id] = {"text": corrected_text, "sc": sc, "cuant_chars": polygon.cuant_chars}

        if manager.update_ocr_results(final_polygons):
            logger.debug(f"Corrección textual completada en: {time.perf_counter() - t0}'s | poligonos restantes: {len(final_polygons) - correced_count}, eliminados: {correced_count}")
            return True
        else:
            logger.warning("Fallo en corrección textual textual")
            return True
        
    def _apply_corrections(self, text: str, semantic_clasification: List[int]) -> str:
        text = text.strip()
        if " " in text:
            if bool(_bad_title.fullmatch(text)):
                text = text.replace(" ", "").strip()
                tokens = [text]

            tokens = text.split(' ')
            
        else:
            tokens = [text]

        total_tokens = len(tokens)
        corrected_tokens: List[str] = []

        if not tokens or not total_tokens:
            return ""

        elif total_tokens == 1:
            return self.correct_cases(self._correct_token(text, semantic_clasification[0]))
        
        else:
            for i, token in enumerate(tokens):
                token_sc = semantic_clasification[i]
                corrected_token = self.correct_cases(self._correct_token(token, token_sc))
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

        if len(token) == 1:
            if semantic_clasification == 1 and "0" in token:
                return token.replace("0", "O")
            return token

        if token.isdecimal():
            return token

        if semantic_clasification in (4, 5):
            return self._correct_cuants(token)
            
        if semantic_clasification in (1, 2):
            if token.endswith("m1"):
                token = token.replace("1", "l")
                
            token = correct_subfix(token)
            
            if semantic_clasification == 1 and "0" in token:
                token = token.replace("0", "O")
                
            if len(token) > 3 and token.isalpha() and token.endswith("Q"):
                token = token.replace("Q", "O")

            return find_umd(token)
            
        # if token.isalpha():
        #     token = correct_subfix(token)
        #     return token
        if semantic_clasification == 3:
            if token.endswith("O"):
                if token.startswith("1") :
                    token = token.replace("O", "0")

                elif token.startswith("C7"):
                    token = token.replace("7", "/")
                    token = token.replace("O", "0")
            return token
            
        else:
            return token
    
    def _correct_cuants(self, token: str) -> str:
        corrected_chars = [numeric_corrections.get(ch, ch) for ch in token]
        return ''.join(corrected_chars)

    def correct_cases(self, text: str) -> str:
        if not text:
            return ""
        elif text.isdecimal():
            return text
        elif text.isupper() or text.islower() or text.istitle():
            return text
        else:
            total_text = len(text)
            total_uppers = (text.count(ascii_uppercase) / total_text)
            total_lowers = (text.count(ascii_lowercase) / total_text)
            if total_uppers > total_lowers or total_lowers < 0.1:
                return text.upper()
            elif total_lowers > total_uppers or total_uppers < 0.1:
                return text.lower()
            else:
                return text.title()