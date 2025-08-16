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
        self.min_word_len_for_frag = self.config.get("min_word_len_for_frag", 2)

    def clean_and_analyze_batch(
        self, 
        polygon_ids: List[str],
        batch_result: List[Optional[Dict[str, Any]]]
    ) -> Tuple[List[Optional[Dict[str, Any]]], List[Tuple[str, int]]]:
        """
        Procesa un lote de resultados de OCR. Limpia el texto y devuelve una
        lista de IDs de polígonos que necesitan ser re-evaluados para fragmentación.

        Args:
            polygon_ids: Lista de IDs de polígonos correspondientes a cada resultado.
            batch_result: Lista de resultados de PaddleOCR (dict con 'text' y 'confidence').

        Returns:
            Un tuple conteniendo:
            - La lista de resultados de OCR con el texto ya limpiado.
            - Una lista de tuplas (poly_id, cantidad_de_fragmentos_necesarios)
        """
        if len(batch_result) != len(polygon_ids):
            logger.error("La cantidad de resultados de OCR no coincide con la de IDs de polígonos.")
            return [], []

        cleaned_batch_results: List[Optional[Dict[str, Any]]] = []
        fragmentation_suggestions: List[Tuple[str, int]] = []

        for result, poly_id in zip(batch_result, polygon_ids):
            if not result or not result.get("text"):
                cleaned_batch_results.append(result)
                continue

            original_text = result["text"]
            
            # 1. Limpiar el texto
            cleaned_text = self._process_single_text(original_text)
            
            # 2. Analizar si necesita fragmentación
            word_count = len(cleaned_text.split())
            if word_count > 1:
                fragmentation_suggestions.append((poly_id, word_count))

            # Mantener la estructura original del dict
            cleaned_result = result.copy()
            cleaned_result["text"] = cleaned_text
            cleaned_batch_results.append(cleaned_result)
        
        return cleaned_batch_results, fragmentation_suggestions

    def _process_single_text(self, text: str) -> str:
        """
        Limpia una única cadena de texto, aplicando un tratamiento diferenciado
        y seguro a los valores que parecen numéricos.
        """
        if not isinstance(text, str):
            return (text) if text is not None else ""

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
                    clean_all=False, extra_spaces=True, stemming= False,
                    stopwords= True,
                    lowercase= False,
                    numbers = False,
                    punct = False,
                    reg = '',
                    reg_replace = '',
                    stp_lang= 'spanish'
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