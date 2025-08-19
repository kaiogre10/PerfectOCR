# PerfectOCR/core/workers/ocr/text_cleaner.py
import logging
import re
from typing import Dict, Any, List, Optional
import cleantext
from core.domain.data_formatter import DataFormatter
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
        
        polygons: Dict[str, Any] = manager.get_polygons()
        polygon_ids = list(polygons.keys())
        
        final_results: List[Optional[Dict[str, Any]]] = self.clean_and_analyze_batch(polygons)
        
        return manager.update_ocr_results(final_results, polygon_ids)
                 
    def clean_and_analyze_batch( 
        self,
        polygons: Dict[str, Any]
    ) -> List[Optional[Dict[str, Any]]]:
        logger.debug("Limpieza de texto iniciada")

        polygon_ids = list(polygons.keys())
        final_results = [
            {
                "text": polygons[pid].get("ocr_text", ""),
                "confidence": polygons[pid].get("ocr_confidence", 0.0)
            }
            for pid in polygon_ids
        ]

        cleaned_batch_results: List[Optional[Dict[str, Any]]] = []
        false_count = 0
        false_polygons: List[Dict[str, Any]] = []

        for result, poly_id in zip(final_results, polygon_ids):
            cleaned_result = result.copy() if result else {}
            text = cleaned_result.get("text", "").strip()
            confidence = cleaned_result.get("confidence", 0.0)

            # Condiciones para marcar como inválido
            if (
                not text or
                confidence < 70.0 or
                re.fullmatch(r'[\s\.\-_,;:]+', text)
            ):
                cleaned_result["status"] = False
                cleaned_result["text"] = ""
                cleaned_result["confidence"] = 0.0
                false_count += 1
                false_polygons.append({"id": poly_id, "text": text, "confidence": confidence})
            else:
                cleaned_result["status"] = True
                cleaned_result["text"] = self._process_single_text(text)

            cleaned_batch_results.append(cleaned_result)
        
        if false_count > 0:
            logger.info(f"Polígonos marcados como status=False: {false_count}")
            for fp in false_polygons:
                logger.info(f"ID: {fp['id']} | Texto: '{fp['text']}' | Confianza: {fp['confidence']}")
        else:
            logger.info("Sin polígonos problemáticos")
            
        return cleaned_batch_results

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
            if self._is_likely_numeric_or_code(token):
                # --- RUTA DE ALTA SEGURIDAD PARA NÚMEROS ---
                safe_word = self._safe_normalize_numeric_separators(token)
                processed_words.append(safe_word)
            else:
                # --- RUTA DE LIMPIEZA GENERAL PARA TEXTO ---
                cleaned_word = cleantext.clean(
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
                processed_words.append(cleaned_word)
        
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

    # def get_cleaning_stats(original_text: str, cleaned_text: str) -> Dict[str, Any]:
    #     """Obtiene estadísticas de la limpieza aplicada."""
    #     return {
    #         'original_length': len(original_text),
    #         'cleaned_length': len(cleaned_text),
    #         'text_changed': original_text != cleaned_text,
    #         'numeric_integrity_enforced': True,
    #         'cleaning_type': 'high_safety_garbage_removal',
    #         'library_used': 'clean-text (conditional)'
    #         }