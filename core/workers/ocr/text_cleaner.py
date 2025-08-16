# PerfectOCR/core/workers/ocr/text_cleaner.py
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from cleantext import clean

logger = logging.getLogger(__name__)

class TextCleaner:
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
        self.project_root = project_root
        # Configuración para la limpieza, puedes añadir umbrales aquí
        self.config = config.get("text_cleaner", {})
        self.min_word_len_for_frag = self.config.get("min_word_len_for_frag", 3)

    def _clean_and_analyze_batch(
        self, 
        batch_result: List[Optional[Tuple[str, float]]], 
        polygon_ids: List[str]
    ) -> Tuple[List[Optional[Tuple[str, float]]], List[str]]:
        """
        Procesa un lote de resultados de OCR. Limpia el texto y devuelve una
        lista de IDs de polígonos que necesitan ser re-evaluados para fragmentación.

        Args:
            batch_result: Lista de resultados de PaddleOCR (tupla de texto y confianza).
            polygon_ids: Lista de IDs de polígonos correspondientes a cada resultado.

        Returns:
            Un tuple conteniendo:
            - La lista de resultados de OCR con el texto ya limpiado.
            - Una lista de IDs de polígonos que se sugiere fragmentar.
        """
        if len(batch_result) != len(polygon_ids):
            logger.error("La cantidad de resultados de OCR no coincide con la de IDs de polígonos.")
            return [], []

        cleaned_batch_results = []
        fragmentation_candidates = []

        for result, poly_id in zip(batch_result, polygon_ids):
            if not result or not result[0]:
                cleaned_batch_results.append(result)
                continue

            original_text = result[0]
            confidence = result[1]

            # 1. Limpiar el texto
            cleaned_text = self._process_single_text(original_text)
            
            # 2. Analizar si necesita fragmentación
            if self._text_needs_fragmentation(cleaned_text):
                logger.debug(f"Polígono '{poly_id}' marcado para fragmentación (Texto: '{cleaned_text}')")
                fragmentation_candidates.append(poly_id)

            cleaned_batch_results.append((cleaned_text, confidence))
        
        return cleaned_batch_results, fragmentation_candidates

    def _process_single_text(self, text: str) -> str:
        """
        Limpia una única cadena de texto, aplicando un tratamiento diferenciado
        y seguro a los valores que parecen numéricos.
        """
        if not isinstance(text, str):
            return str(text) if text is not None else ""

        # Dividir por espacios para procesar token por token, preservando la estructura.
        words = text.split(' ')
        processed_words = []

        for word in words:
            if self._is_likely_numeric_or_code(word):
                # --- RUTA DE ALTA SEGURIDAD PARA NÚMEROS ---
                safe_word = self._safe_normalize_numeric_separators(word)
                processed_words.append(safe_word)
            else:
                # --- RUTA DE LIMPIEZA GENERAL PARA TEXTO ---
                cleaned_word = clean(
                    word,
                    fix_unicode=True, to_ascii=False, lower=False,
                    no_line_breaks=True, no_urls=True, no_emails=True,
                    no_phone_numbers=True, no_numbers=False, no_digits=False,
                    no_currency_symbols=False, no_punct=False, lang="es"
                )
                processed_words.append(cleaned_word)
        
        return ' '.join(processed_words)

    def _text_needs_fragmentation(self, text: str) -> bool:
        """
        Determina si el texto de un polígono sugiere que debería ser fragmentado.
        La lógica es simple: si contiene más de una "palabra" significativa.
        """
        # Dividir por espacios y filtrar palabras muy cortas que pueden ser ruido
        meaningful_words = [
            word for word in text.split() if len(word) >= self.min_word_len_for_frag
        ]
        return len(meaningful_words) > 1

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
        return re.sub(rf"(?<=\d)[{symbols_to_dot}](?=\d)", ".", token)

    def get_cleaning_stats(self, original_text: str, cleaned_text: str) -> Dict[str, Any]:
        """Obtiene estadísticas de la limpieza aplicada."""
        return {
            'original_length': len(original_text),
            'cleaned_length': len(cleaned_text),
            'text_changed': original_text != cleaned_text,
            'numeric_integrity_enforced': True,
            'cleaning_type': 'high_safety_garbage_removal',
            'library_used': 'clean-text (conditional)'
            }