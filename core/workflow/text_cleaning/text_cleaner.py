# PerfectOCR/core/text_cleaning/text_cleaner.py
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from cleantext import clean

logger = logging.getLogger(__name__)

class TextCleaner:
    """
    Limpiador de texto de alta seguridad para ruido OCR, con un enfoque
    radicalmente conservador para valores numéricos.
    - NO corrige palabras.
    - NO elimina dígitos bajo ninguna circunstancia.
    - Preserva el espaciado para mantener la geometría.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        logger.info("TextCleaner (High-Safety) inicializado.")

    def _is_likely_numeric_or_code(self, token: str) -> bool:
        """
        Determina si un token es probablemente un número, moneda o código.
        Es muy inclusivo para evitar la pérdida de datos.
        """
        if not token:
            return False
        # Si contiene al menos un dígito, trátalo con cuidado.
        if re.search(r'\d', token):
            return True
        # Considera también valores monetarios sin dígitos como '$'
        if token in ['$','€','£']:
            return True
        # Patrones monetarios más completos
        monetary_patterns = [
            r'^\$?\d+(\.\d+)?$',              # $100 o 100
            r'^\$?\d{1,3}(,\d{3})*(\.\d+)?$', # $1,000.00
            r'^\d+[.,]\d+[.,]\d+$',           # 1.000.000
        ]
        
        for pattern in monetary_patterns:
            if re.match(pattern, token):
                return True
        return False

    def _safe_normalize_numeric_separators(self, token: str) -> str:
        """
        Normaliza DE FORMA SEGURA los separadores en un token numérico.
        Solo reemplaza caracteres conocidos por puntos, NUNCA elimina dígitos.
        """
        # Símbolos comunes de OCR que deben ser puntos decimales.
        # Incluye comilla simple, invertida, acentos, y la coma.
        symbols_to_dot = r"`'´,"
        
        # El patrón busca un dígito, seguido de uno de los símbolos, seguido de otro dígito.
        # Esto evita cambiar comas en texto normal (ej. "hola,mundo").
        return re.sub(rf"(?<=\d)[{symbols_to_dot}](?=\d)", ".", token)

    def clean_text(self, text: str, confidence: Optional[float] = None) -> str:
        """
        Limpia el texto aplicando un tratamiento diferenciado y seguro a los
        valores que parecen numéricos para evitar cualquier pérdida de datos.
        """
        if not isinstance(text, str):
            return str(text) if text is not None else ""

        original_text = text
        
        # Dividir por espacios para procesar token por token, preservando la estructura.
        words = original_text.split(' ')
        processed_words = []

        for word in words:
            if self._is_likely_numeric_or_code(word):
                # --- RUTA DE ALTA SEGURIDAD PARA NÚMEROS ---
                # Solo se normalizan los separadores. No hay más limpieza.
                safe_word = self._safe_normalize_numeric_separators(word)
                processed_words.append(safe_word)
            else:
                # --- RUTA DE LIMPIEZA GENERAL PARA TEXTO ---
                # Este texto no parece numérico, se puede limpiar con clean-text.
                cleaned_word = clean(
                    word,
                    fix_unicode=True,
                    to_ascii=False,
                    lower=False,
                    no_line_breaks=True,
                    no_urls=True,
                    no_emails=True,
                    no_phone_numbers=True,
                    no_numbers=False,       # No eliminar números (doble seguridad)
                    no_digits=False,        # No eliminar dígitos (doble seguridad)
                    no_currency_symbols=False,
                    no_punct=False,
                    lang="es"
                )
                processed_words.append(cleaned_word)

        # Reconstruir la línea con el espaciado original.
        cleaned_text = ' '.join(processed_words)

        if cleaned_text != original_text:
            logger.debug(f"Texto limpiado (seguro): '{original_text[:70]}...' -> '{cleaned_text[:70]}...'")
        
        return cleaned_text

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