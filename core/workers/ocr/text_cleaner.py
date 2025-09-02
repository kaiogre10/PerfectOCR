# PerfectOCR/core/workers/ocr/text_cleaner.py
import logging
import re
from typing import Dict, Any, List, Optional
from cleantext import clean # type: ignore
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons
from core.factory.abstract_worker import OCRAbstractWorker

logger = logging.getLogger(__name__)

class TextCleaner(OCRAbstractWorker):
    """
    Limpiador de texto de alta seguridad para ruido OCR y analizador de contenido.
    - Limpia el texto de forma conservadora, protegiendo datos numéricos.
    - Identifica polígonos que contienen múltiples palabras, marcándolos como
      candidatos para una futura fragmentación geométrica.
    - NO corrige palabras.
    - NO elimina dígitos bajo ninguna circunstancia.
    - Preserva el espaciado para mantener la geometría.
    """
    
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.config = config
        self.config = config.get("text_cleaner", {})
        self.min_word_len_for_frag = self.config.get("min_confidence", 70.0)
                    
    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
        polygon_ids = list(polygons.keys())
        
        logger.info(f"Polígonos de entrada: {len(polygon_ids)}")
        
        # Filtrar y eliminar basura directamente
        valid_results: List[Optional[Dict[str, Any]]] = []
        valid_polygon_ids: List[str] = []
        eliminated_count = 0
        
        # Iterar sobre una lista de claves para evitar modificar el dict durante iteración
        for poly_id in list(polygons.keys()):
            polygon = polygons[poly_id]
            text = polygon.ocr_text or ""
            confidence = polygon.ocr_confidence or 0.0
            
            # Criterio de basura
            if (
                not text or
                (confidence < 70.0 and not self._is_likely_numeric_or_code(text)) or
                re.fullmatch(r'[\s\.\-_,;:]+', text)
            ):
                # Eliminar y las dataclasses
                if manager.workflow and poly_id in manager.workflow.polygons:
                    del manager.workflow.polygons[poly_id]
                eliminated_count += 1
                logger.debug(f"Eliminado: ID: {poly_id} | Texto: '{text}' | Confianza: {confidence}")
            else:
                # Conservar válido
                result: Dict[str, Any] = {
                    "text": self._process_single_text(text),
                    "confidence": confidence,
                    "status": True
                }
                valid_results.append(result)
                valid_polygon_ids.append(poly_id)
        
        logger.info(f"Polígonos eliminados: {eliminated_count}")
        logger.info(f"Polígonos de salida: {len(valid_results)}")
        
        final_results = valid_results
        polygon_ids = valid_polygon_ids
        return manager.update_ocr_results(final_results, polygon_ids)
                 
    def _process_single_text(self, text: str) -> str:
        """
        Limpia una única cadena de texto, aplicando un tratamiento diferenciado
        y seguro a los valores que parecen numéricos.
        """
        if not isinstance(text, str):
            return (text) if text is not None else ""
            
        # Dividir por espacios para procesar token por token, preservando la estructura.
        words = text.split(' ')
        processed_words: List[str] = []

        for token in words:
            if not token.strip():  # Evitar procesar tokens vacíos
                processed_words.append(token)
            else:
                # Limpieza de caracteres ANTES de decidir si es numérico
                cleaned_token = self._clean_characters_in_word(token)
                
                if self._is_likely_numeric_or_code(cleaned_token):
                    # --- RUTA DE ALTA SEGURIDAD PARA NÚMEROS ---
                    safe_word = self._safe_normalize_numeric_separators(cleaned_token)
                    processed_words.append(safe_word)
                else:
                    try:
                        cleaned_token = clean(
                            token,
                            clean_all=False, extra_spaces=True, stemming= False,
                            stopwords= True,
                            lowercase= False,
                            numbers = False,
                            punct = False,
                            reg = '',
                            reg_replace = '',
                            stp_lang= 'spanish'
                        )
                        processed_words.append(cleaned_token)
                    except Exception:
                        processed_words.append(cleaned_token)
        
        return ' '.join(processed_words)
        
        
    # def _text_needs_fragmentation(text: str) -> bool:
    #     """
    #     Determina si el texto de un polígono sugiere que debería ser fragmentado.
    #     La lógica es simple: si contiene más de una "palabra" significativa.
    #     """
    #     # Dividir por espacios y filtrar palabras muy cortas que pueden ser ruido
    #     meaningful_words = [
    #         word for word in text.split() if len(word) >= min_word_len_for_frag
    #     ]
    #     return len(meaningful_words) > 1

    def _is_likely_numeric_or_code(self, token: str) -> bool:
        """
        Determina si un token es probablemente un número, moneda o código.
        Es muy inclusivo para evitar la pérdida de datos.
        """
        if not token:
            return False
        if re.search(r'\d', token):
            return True
        if token in ['$','€','£']:
            return True
        monetary_patterns = [
            r'^\$?\d+(\.\d+)?$',
            r'^\$?\d{1,3}(,\d{3})*(\.\d+)?$',
            r'^\d+[.,]\d+[.,]\d+$',
        ]
        for pattern in monetary_patterns:
            if re.match(pattern, token):
                return True
        return False

    def _safe_normalize_numeric_separators(self, token: str) -> str:
        """
        Normaliza DE FORMA SEGURA los separadores en un token numérico.
        """
        symbols_to_dot = r"`'´,"
        safe_word = re.sub(rf"(?<=\d)[{symbols_to_dot}](?=\d)", ".", token)
        return safe_word

    def _clean_characters_in_word(self, token: str) -> str:
        """Limpieza de caracteres específicos en palabras individuales."""
        if not token:
            return token
        
        # Reemplazar caracteres confusos OCR
        char_replacements = {
            # '0': 'O',  # Cero confundido con O mayúscula
            # '1': 'l',  # Uno confundido con l minúscula
            # '5': 'S',  # Cinco confundido con S mayúscula
            # '8': 'B',  # Ocho confundido con B mayúscula
            # '|': 'I',  # Barra confundida con I mayúscula
            # '!': '1',  # Exclamación confundida con uno
            # 'G': '6',  # G mayúscula confundida con seis
            # 'g': '9',  # g minúscula confundida con nueve
            # 'Z': '2',  # Z mayúscula confundida con dos
            # 'z': '2',  # z minúscula confundida con dos
        }
        
        # Aplicar reemplazos
        if char_replacements is not None:
            for wrong_char, correct_char in char_replacements.items():
                token = token.replace(wrong_char, correct_char)
            else:
                pass
        
        # Limpiar caracteres de ruido OCR (mantener solo alfanuméricos + símbolos útiles)
        token = re.sub(r'[^\w\s\.\,]', '', token)
        
        return token