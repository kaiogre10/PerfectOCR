# PerfectOCR/core/workers/ocr/text_cleaner.py
import logging
import time
from typing import Dict, Any, List
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons
from core.factory.abstract_worker import OCRAbstractWorker
from utils.text_utils import remove_special_sequences, punct_strip, separate_punt, is_acronym
from utils.compiled_utils import validate_text

logger = logging.getLogger(__name__)

class TextCleaner(OCRAbstractWorker):
    """
    Limpiador de texto de alta seguridad para ruido OCR y analizador de contenido.
    - Limpia el texto de forma conservadora, protegiendo datos numéricos.
    - Identifica polígonos que contienen múltiples palabras y los fragmenta geométricamente si hay suficiente evidencia visual (contornos).
    - NO corrige palabras.
    - NO elimina dígitos bajo ninguna circunstancia.
    - Preserva el espaciado para mantener la geometría.
    """
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.del_output_log = config.get("text_del")
        self.output_log = config.get("text_clean")
                    
    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        worker_name = context.get("worker_name") or "paddle_wrapper"
        logger.debug(f"Inicia cleanner")
        t0 = time.perf_counter()
        if not manager.workflow or not manager.workflow.polygons:
            logger.warning("TextCleaner: No hay polígonos en el workflow para procesar.")
            return False

        polygons_in: Dict[str, Polygons] = manager.workflow.polygons

        logger.debug(f"Cantidad de polígonos recibidos:{len(polygons_in)}")
        final_polygons: Dict[str, Dict[str, Any]] = {}
        eliminated_count = 0

        for poly_id, polygon in polygons_in.items():
            kf = polygon.key_field or None
            sc = polygon.semantic_clasification
            text = polygon.ocr_text or ""

            if 0 in sc and kf is not None:
                # logger.debug(f"'{poly_id}' con KEYFIELD ya no se limpia: '{polygon.ocr_text}'")
                final_polygons[poly_id] = {"text": text}
                continue
                
            if not text or not validate_text(text):
                if self.del_output_log:
                    logger.info(f"Eliminado {poly_id} sin texto valido incial: '{text}'")
                eliminated_count += 1
                continue

            if is_acronym(text):
                final_polygons[poly_id] = {"text": text}
                continue

            text_sec = remove_special_sequences(text)
               
            if not text_sec or not validate_text(text_sec):
                if self.del_output_log:
                    logger.info(f"{poly_id} sin texto valido después de secuencias especiales: '{text}'")
                eliminated_count += 1
                continue

            sep_text = separate_punt(text_sec)

            if not sep_text or not validate_text(sep_text):
                if self.del_output_log:
                    logger.info(f"{poly_id} sin texto válido después de eliminar puntuaciones '{text_sec}'")
                eliminated_count += 1
                continue

            txt = self.process_single_text(sep_text, polygon)
            if not txt or not validate_text(txt):
                eliminated_count += 1
                if self.del_output_log:
                    logger.info(f"{poly_id} sin texto válido después de espacios: '{sep_text}'")
                continue
            
            else:
                final_polygons[poly_id] = {"text": txt}
                if self.output_log and txt != text:
                    logger.info(f"Limpieza de '{poly_id}' | Original: '{text}' | Ruido:'{set(text.split(" ")).difference(set(txt.split(" ")))}' → '{txt}'")

        if manager.update_ocr_results(final_polygons, worker_name):
            logger.debug(f"Limpieza textual completada en: {time.perf_counter() - t0}'s | poligonos restantes: {len(final_polygons) - eliminated_count}, eliminados: {eliminated_count}")
            return True
        else:
            logger.warning("Fallo en limpieza textual")
            return True

    def process_single_text(self, text: str, polygon: Polygons) -> str:
        """
        Limpia una única cadena de texto, aplicando un tratamiento diferenciado
        y seguro a los valores que parecen numéricos.
        """ 
        text = text.strip()
        if not text:
            return ""

        # Dividir por espacios para procesar token por token, preservando la estructura.
        words = text.split(' ')
        processed_words: List[str] = []

        for token in words:
            if token.isalnum() or token.isalpha() or token.isdecimal():
                processed_words.append(token)
                continue

            clean_token = punct_strip(token)
                # Eliminar tokens que sean un carácter especial especificado (ej. ")")
            if not clean_token or not any(c.isalnum() for c in clean_token):
                if self.del_output_log:
                    logger.info(f"Eliminado texto basura : '{clean_token}' in {polygon.polygon_id if polygon else ' '}")
                continue
            else:
                processed_words.append(clean_token)
                continue
        
        return ' '.join(processed_words).strip()
