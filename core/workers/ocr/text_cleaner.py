# PerfectOCR/core/workers/ocr/text_cleaner.py
import logging
import re
import dataclasses
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
    - Identifica polígonos que contienen múltiples palabras y los fragmenta
      geométricamente si hay suficiente evidencia visual (contornos).
    - NO corrige palabras.
    - NO elimina dígitos bajo ninguna circunstancia.
    - Preserva el espaciado para mantener la geometría.
    """
    
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get("text_cleaner", {})        
        # Lista de caracteres especiales a eliminar si aparecen solos.
        # Configurada directamente en el worker en lugar de leerla desde un YAML.
        self.chars = [
            ")", "(", "]", "[", "{", "}", "|", "*", "^", "#", "@",
            "-", "~", "_", "+", "=", "<", ">", ";", ":",
            "'", "!", "¡", "?", "¿", "'", "/", "\\"
        ]

        # normalizar a conjunto de caracteres de longitud 1
        self.drop_single_chars = set(c for c in self.chars if isinstance(c, str) and len(c) == 1)
                    
    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        self.min_confidence: float  = self.worker_config.get("min_confidence")

        if not manager.workflow or not manager.workflow.polygons:
            logger.warning("TextCleaner: No hay polígonos en el workflow para procesar.")
            return True

        polygons_in: Dict[str, Polygons] = manager.workflow.polygons
        list_of_final_polygons: List[Polygons] = []
        
        sorted_poly_ids = sorted(
            polygons_in.keys(), 
            key=lambda p_id: (polygons_in[p_id].geometry.centroid[1], polygons_in[p_id].geometry.centroid[0])
        )
        eliminated_count = 0
        for poly_id in sorted_poly_ids:
            polygon = polygons_in[poly_id]
            text = polygon.ocr_text or ""
            confidence = polygon.ocr_confidence or 0.0
            semantic_clasification = polygon.semantic_clasification or ""

        
            if self._is_polygon_single_special(text):
                logger.debug(f"Eliminado {poly_id} unico:'{text}'")
                continue
            
            text = self._filter_low_prob_tokens(text, polygon, manager)
            text = text or ""

            text = self._remove_special_chars(text)

            if (not text.strip() or
                (confidence < self.min_confidence and polygon.semantic_clasification not in ("numeric", "quantitative", "rfc", "umd")) or
                re.fullmatch(r'[\s\.\-_,;:]+', text)):
                logger.debug(f"Eliminado {poly_id}: | Texto: {text}, conf: {confidence}")
                continue

            # 3. Ruta Normal: Limpiar texto del polígono vacío
            cleaned_text = self._process_single_text(text, polygon, semantic_clasification, manager)
            if cleaned_text:
                updated_polygon = dataclasses.replace(polygon, ocr_text=cleaned_text, was_refined=True)
                list_of_final_polygons.append(updated_polygon)
            else:
                logger.debug(f"Eliminado {poly_id}: | Texto: '{text}'")

        eliminated_count += 1

        # 4. Reconstrucción y reindexación final
        final_polygons_dict: Dict[str, Polygons] = {}
        for idx, poly_obj in enumerate(list_of_final_polygons):
            new_id = f"poly_{idx:04d}"
            final_poly_obj = dataclasses.replace(poly_obj, polygon_id=new_id, was_refined=True)
            final_polygons_dict[new_id] = final_poly_obj
            
            # 5. Reemplazo directo en el manager
        manager.workflow.polygons = final_polygons_dict

        logger.debug(f"{eliminated_count} eliminados. Total final: {len(final_polygons_dict)}")
            
        return True

    def _process_single_text(self, text: str, polygon: Polygons, semantic_clasification :str, manager: DataFormatter) -> str:
        """
        Limpia una única cadena de texto, aplicando un tratamiento diferenciado
        y seguro a los valores que parecen numéricos.
        """ 
        
        # Dividir por espacios para procesar token por token, preservando la estructura.
        words = text.split(' ')
        processed_words: List[str] = []

        for token in words:
            if not token.strip():  # Evitar procesar tokens vacíos
                processed_words.append(token)
                continue
            
            # Eliminar tokens que sean un carácter especial especificado (ej. ")")
            if self._is_stray_single_special(token):
                logger.debug(f"Eliminado unico: '{token}' in {polygon.polygon_id if polygon else ''}")
                continue
            
            if self._is_likely_numeric(token, semantic_clasification):
                processed_words.append(token)
            else:
                processed_words.append(token)
        
        return ' '.join(processed_words)
        
    def _is_likely_numeric(self, token: str, semantic_clasification: str) -> bool:
        if semantic_clasification in ("numeric", "quantitative"):
            return True
        return False

    def _filter_low_prob_tokens(self, text: str, polygon: Polygons, manager: DataFormatter) -> str:
        min_char = int(self.worker_config.get("min_char"))
        min_probability = float(self.worker_config.get("min_probability"))
        if polygon.ocr_confidence and polygon.ocr_confidence >= self.min_confidence:
            return text
            
        try:
            if getattr(polygon, "semantic_clasification", None) in ("numeric", "quantitative"):
                return text

            tokens = text.split(' ')
            kept: List[str] = []
            removed = 0
            total = 0
            for tok in tokens:
                t = tok.strip()
                if not t:
                    kept.append(tok)
                    continue

                total += 1

                if any(ch.isdigit() for ch in t):
                    kept.append(tok)
                    continue

                eff_len = len(''.join(ch for ch in t if not ch.isspace()))
                if eff_len <= min_char:
                    score = self._token_freq_score(t, manager)
                    
                    if score < min_probability:
                        removed += 1
                        logger.debug(f"Eliminado:{polygon.polygon_id} | Texto:'{t}' | Probabilidad: {score:.4f}")
                        continue
                    kept.append(tok)
                else:
                    kept.append(tok)

            out = ' '.join(kept)
            if removed > 0:
                logger.debug(f"{polygon.polygon_id} | Texto: '{text}' => '{out}'")
            return out
        
        except Exception as e:
            logger.error(f"Error eliminando tokens por frecuencia: {e}", exc_info=True)

            return text

    def _normalize_char_for_freq(self, ch: str, manager: DataFormatter) -> str:
         # Mantén tildes/ñ si existen en la tabla; si no, haz fallback a su base
        if ch in self._get_frecuency_norm(manager): #type: ignore
            return ch
        base_map = {
            "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
            "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U",
            "ü": "u", "Ü": "U", "ñ": "n", "Ñ": "N",
        }
        return base_map.get(ch, ch)

    def _is_stray_single_special(self, token: str) -> bool:
        """
        True si el token (tras strip) es exactamente un carácter y está en la lista
        configurada de caracteres a eliminar cuando aparecen aislados.
        """
        t = token.strip()
        return len(t) == 1 and t in self.drop_single_chars
    
    def _is_polygon_single_special(self, text: str) -> bool:
        """
        True si el texto del polígono (tras strip) es exactamente un carácter
        y ese carácter está en la lista drop_single_chars.
        (Esta verificación ignora la confianza del OCR.)
        """
        if not text:
            return False
        t = text.strip()
        return len(t) == 1 and t in self.drop_single_chars

    def _get_frecuency_norm(self, manager: DataFormatter) -> Optional[Dict[str, float]]:

        try:
            if not manager or not getattr(manager, "workflow", None):
                logger.error("Manager o workflow ausente")
                return None

            frecuency_char_raw = manager.get_frecuency_char()
            if frecuency_char_raw is None:
                logger.error("get_frecuency_char() devolvió None")
                return None
            frecuency_char: Dict[str, int] = frecuency_char_raw

            max_val = float(max(frecuency_char.values()))

            freq_norm: Dict[str, float] = {char: (val / max_val) * 100 for char, val in frecuency_char.items()}
            return freq_norm
        
        except Exception as e:
            logger.error(f"Error al obtener frecuencias normalizadas: {e}", exc_info=True)

    def _token_freq_score(self, token: str, manager: DataFormatter) -> float:
        freq_norm = self._get_frecuency_norm(manager)
        if not freq_norm:
            return 100.0  # si no hay tabla, no castigues
        
        letters: List[float] = []
        for ch in token:
            if ch.isalpha():
                norm = self._normalize_char_for_freq(ch.lower(), manager)
                if norm in freq_norm:
                    letters.append(freq_norm[norm])
                elif norm.isalpha():
                    letters.append(0.0)
        if not letters:
        
            return 100.0  # tokens sin letras no se filtran por frecuencia
        return sum(letters) / float(len(letters))

    def _remove_special_chars(self, text: str) -> str:
        """
        Elimina todos los caracteres especiales, tanto solitarios como en secuencia.
        Preserva dígitos, letras y espacios.
        """
        if not text:
            return text

        special_chars = self.drop_single_chars
        if not special_chars:
            logger.warning("Usando patron regex")
            pattern = r'[^A-Za-z0-9\s$¢.,\/\\]'
        else:
            # escapamos los caracteres especiales para regex
            chars_escaped = re.escape("".join(special_chars))
            pattern = r'[' + chars_escaped + r']'

        cleaned = re.sub(pattern, '', text)
        return cleaned
