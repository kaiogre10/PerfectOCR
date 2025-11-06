# PerfectOCR/core/workers/ocr/text_cleaner.py
import logging
import re
import dataclasses
from typing import Dict, Any, List
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons
from core.factory.abstract_worker import OCRAbstractWorker
from core.utils.text_validator import validate_text, validate_alone_chars, special_chars, get_char_num

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
        self.worker_config = config.get("text_cleaner", {})
        self.min_confidence: float  = self.worker_config.get("min_confidence")
        self.min_char = int(self.worker_config.get("min_char"))
        self.min_probability = float(self.worker_config.get("min_probability"))
        self.char_num: List[str] = get_char_num()
        self.chars: List[str] = special_chars()
        self.drop_single_chars = set(c for c in self.chars if isinstance(c, str) and len(c) == 1) # type: ignore
                    
    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        if not manager.workflow or not manager.workflow.polygons:
            logger.warning("TextCleaner: No hay polígonos en el workflow para procesar.")
            return True

        polygons_in: Dict[str, Polygons] = manager.workflow.polygons        
        sorted_poly_ids = sorted(
            polygons_in.keys(), 
            key=lambda p_id: (polygons_in[p_id].geometry.centroid[1], polygons_in[p_id].geometry.centroid[0])
        )

        logger.debug(f"Cantidad de polígonos recibidos:{len(sorted_poly_ids)}")
        list_of_final_polygons: List[Polygons] = []
        eliminated_count = 0

        for poly_id in sorted_poly_ids:
            polygon = polygons_in[poly_id]
            confidence = polygon.ocr_confidence or 0.0
            sc: List[int] | int = polygon.semantic_clasification or 0
            text = polygon.ocr_text or ""

            if not validate_text(text):
                logger.debug(f"Eliminado {poly_id} sin texto inicial")
                eliminated_count += 1
                continue

            if not validate_alone_chars(text):
                logger.info(f"Eliminado {poly_id} por soledad: '{text}'")
                eliminated_count += 1
                continue
            
            fil_text = self._filter_low_prob_tokens(text, polygon, manager)
            if not validate_text(fil_text):
                logger.debug(f"Eliminado {poly_id} sin texto después de filtrado de probabilidad")
                eliminated_count += 1
                continue

            text = self._remove_special_chars(text)

            is_numeric_like = (isinstance(sc, list) and any(c in [1, 2, -2] for c in sc)) or \
                            (isinstance(sc, int) and sc in [1, 2, -2])

            if (not validate_text(text) or
                (confidence < self.min_confidence and not is_numeric_like) or re.fullmatch(r'[\s\.\-_,;:]+', text)):
                reason = "sin texto" if not validate_text(text) \
                    else f"baja confianza ({confidence:.2f})" if confidence < self.min_confidence and not is_numeric_like \
                    else "solo caracteres de puntuación"
                logger.debug(f"Eliminado {poly_id}: '{text}' (Razón: {reason})")
                eliminated_count += 1
                continue

            # 3. Ruta Normal: Limpiar texto del polígono vacío
            cleaned_text = self._process_single_text(text, polygon)
            if validate_text(cleaned_text):
                updated_polygon = dataclasses.replace(polygon, ocr_text=cleaned_text, was_refined=True)
                list_of_final_polygons.append(updated_polygon)
                
            else:
                logger.info(f"Eliminado {poly_id}: Sin texto en limpieza final")
                eliminated_count += 1

        eliminated_count += 1

        # 4. Reconstrucción y reindexación final
        final_polygons_dict: Dict[str, Polygons] = {}
        for idx, poly_obj in enumerate(list_of_final_polygons):
            new_id = f"poly_{idx:04d}"
            final_poly_obj = dataclasses.replace(poly_obj, polygon_id=new_id, was_refined=True)
            final_polygons_dict[new_id] = final_poly_obj
            
            # 5. Reemplazo directo en el manager
        manager.workflow.polygons = final_polygons_dict

        logger.debug(f"{eliminated_count} limpios. Total final: {len(final_polygons_dict)}")
            
        return True

    def _process_single_text(self, text: str, polygon: Polygons) -> str:
        """
        Limpia una única cadena de texto, aplicando un tratamiento diferenciado
        y seguro a los valores que parecen numéricos.
        """ 
        
        # Dividir por espacios para procesar token por token, preservando la estructura.
        words = text.split(' ')
        processed_words: List[str] = []

        for token in words:
            if validate_text(token):  # Evitar procesar tokens vacíos
                processed_words.append(token)
                continue
            
            # Eliminar tokens que sean un carácter especial especificado (ej. ")")
            if self._is_stray_single_special(token):
                logger.info(f"Eliminado unico: '{token}' in {polygon.polygon_id if polygon else ''}")
                continue
        
            else:
                processed_words.append(token)
        
        return ' '.join(processed_words)

    def _filter_low_prob_tokens(self, text: str, polygon: Polygons, manager: DataFormatter) -> str:
        if polygon.ocr_confidence and polygon.ocr_confidence >= self.min_confidence:
            return text
            
        try:
            sc = polygon.semantic_clasification
            is_numeric_like = (isinstance(sc, list) and any(c in [1, 2, -2] for c in sc)) or \
                            (isinstance(sc, int) and sc in [1, 2, -2])
            if is_numeric_like:
                return text

            tokens = text.split(' ')
            kept: List[str] = []
            removed = 0
            total = 0
            for tok in tokens:
                t = tok.strip()
                if not validate_text(t):
                    kept.append(tok)
                    continue

                total += 1

                if any(ch in self.char_num for ch in t):
                    kept.append(tok)
                    continue

                eff_len = len(''.join(ch for ch in t if not ch.isspace()))
                if eff_len < self.min_char:
                    score = self._token_freq_score(t, manager)
                    
                    if score < self.min_probability:
                        removed += 1
                        logger.info(f"Eliminado:{polygon.polygon_id} | Texto:'{t}' | Probabilidad: {score:.4f}")
                        continue
                    kept.append(tok)
                else:
                    kept.append(tok)

            out = ' '.join(kept)
            if removed > 0:
                logger.info(f"{polygon.polygon_id} | Texto: '{text}' => '{out}'")
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
        if not validate_text(text):
            return False
            
        t = text.strip()
        return len(t) == 1 and t in self.drop_single_chars

    def _get_frecuency_norm(self, manager: DataFormatter) -> Dict[str, float]:

        try:
            frecuency_char = manager.get_frecuency_char()
            max_val = max(frecuency_char.values())

            freq_norm: Dict[str, float] = {char: (val / max_val) * 100 for char, val in frecuency_char.items()}
            return freq_norm
        
        except Exception as e:
            logger.error(f"Error al obtener frecuencias normalizadas: {e}", exc_info=True)
            return {}

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
        if not validate_text(text):
            return text

        special_chars = self.drop_single_chars
        if not special_chars:
            logger.warning("Usando patron regex")
            pattern = r'[^A-Za-z0-9\s$¢.,\/\\]'
        else:
            chars_escaped = re.escape("".join(special_chars))
            pattern = r'[' + chars_escaped + r']'

        cleaned = re.sub(pattern, '', text)
        if cleaned != text:
            logger.debug(f"Caracteres especiales eliminados de '{text}' -> '{cleaned}'")
        return cleaned
