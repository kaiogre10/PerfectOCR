# PerfectOCR/core/workers/ocr/text_cleaner.py
import logging
import re
import dataclasses
from typing import Dict, Any, List, Optional
from cleantext import clean # type: ignore
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons
from core.factory.abstract_worker import OCRAbstractWorker
from core.utils.semantic_classifier import is_numeric

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
        self.enabled_outputs = self.config.get("enabled_outputs", {})
        self.output = self.worker_config.get("clean_text", False)
        self.min_confidence = float(self.worker_config.get("min_confidence", {}))
        self.min_char = int(self.worker_config.get("min_char", {}))
        self.min_probability = float(self.worker_config.get("min_probability", {}))
                    
    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        if not manager.workflow or not manager.workflow.polygons:
            logger.warning("TextCleaner: No hay polígonos en el workflow para procesar.")
            return True

        polygons_in: Dict[str, Polygons] = manager.workflow.polygons
        list_of_final_polygons: List[Polygons] = []
        eliminated_count = 0

        # Ordenar polígonos por posición vertical (y luego horizontal) para un procesamiento secuencial
        sorted_poly_ids = sorted(
            polygons_in.keys(), 
            key=lambda p_id: (polygons_in[p_id].geometry.centroid[1], polygons_in[p_id].geometry.centroid[0])
        )
        
        for poly_id in sorted_poly_ids:
            polygon = polygons_in[poly_id]
            text = polygon.ocr_text or ""
            confidence = polygon.ocr_confidence or 0.0
            semantic_type = polygon.semantic_type or ""
    
            text = self._filter_low_prob_tokens(text, polygon, manager)
            text = text or ""

            if (not text.strip() or
                (confidence < self.min_confidence and polygon.semantic_type not in ("numeric", "quantitative")) or
                re.fullmatch(r'[\s\.\-_,;:]+', text)):
                logger.info(f"Eliminado (basura): ID: {poly_id} | Texto: '{text}'")
                continue

            # 3. Ruta Normal: Limpiar texto del polígono vacío
            cleaned_text = self._process_single_text(text, polygon, semantic_type, manager)
            if cleaned_text:
                updated_polygon = dataclasses.replace(polygon, ocr_text=cleaned_text)
                list_of_final_polygons.append(updated_polygon)
            else:
                eliminated_count += 1
                logger.info(f"Eliminado (texto vacío post-limpieza): ID: {poly_id} | Texto: '{text}'")

        # 4. Reconstrucción y reindexación final
        final_polygons_dict: Dict[str, Polygons] = {}
        for idx, poly_obj in enumerate(list_of_final_polygons):
            new_id = f"poly_{idx:04d}"
            final_poly_obj = dataclasses.replace(poly_obj, polygon_id=new_id)
            final_polygons_dict[new_id] = final_poly_obj
        
        # 5. Reemplazo directo en el manager
        manager.workflow.polygons = final_polygons_dict

        if self.output:
            file_name: str = manager.workflow.metadata.image_name
            self._save_json(context, final_polygons_dict, file_name)

        logger.debug(f"Limpieza/{eliminated_count} eliminados. Total final: {len(final_polygons_dict)}")
        return True

    def _process_single_text(self, text: str, polygon: Polygons, semantic_type :str, manager: DataFormatter) -> str:
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

            # Limpieza de caracteres ANTES de decidir si es numérico
            cleaned_token = self._clean_characters_in_word(token)
            
            if self._is_likely_numeric_or_code(cleaned_token, polygon):
                logger.debug(f"Resultados del text_clenanner{cleaned_token}")
            
                processed_words.append(cleaned_token)
            else:
                try:
                    cleaned_token_lib = clean(
                        cleaned_token, # Usar el token ya pre-limpiado
                        clean_all=False, 
                        extra_spaces=True, 
                        stemming=False,
                        stopwords=False, # No eliminar stopwords para no perder contexto
                        lowercase=False,
                        numbers=False,
                        punct=False, # No eliminar puntuación que podría ser relevante
                    )
                    processed_words.append(cleaned_token_lib)
                except Exception:
                    # Si clean-text falla, usar el token pre-limpiado como fallback
                    processed_words.append(cleaned_token)
        
        return ' '.join(processed_words)
        
    def _is_likely_numeric_or_code(self, cleaned_token: str, polygon: Polygons) -> bool:
        """
        Determina si un token es probablemente un número o código
        usando únicamente la clasificación semántica del polígono.
        """
        return is_numeric(cleaned_token, self.config)

    def _filter_low_prob_tokens(self, text: str, polygon: Polygons, manager: DataFormatter) -> str:
        # Si la confianza general del polígono es alta, no tocar sus palabras.
        if polygon.ocr_confidence and polygon.ocr_confidence >= self.min_confidence:
            return text
            
        try:
            if getattr(polygon, "semantic_type", None) in ("numeric", "quantitative"):
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
                if eff_len <= int(getattr(self, "min_char", 2)):
                    score = self._token_freq_score(t, manager)
                    threshold = float(getattr(self, "min_probability", 5.0))
                    if score < threshold:
                        removed += 1
                        logger.info(f"Eliminado (basura): ID: {polygon.polygon_id} | Texto:'{t}' | Score: {score:.2f} ")
                        continue
                    kept.append(tok)
                else:
                    kept.append(tok)

            out = ' '.join(kept)
            if removed > 0:
                logger.info(f"Corrección: ID:={polygon.polygon_id} | Texto: '{text}' => '{out}'")
            return out
        except Exception as e:
            logger.debug(f"Error eliminando tokens por frecuencia: {e}", exc_info=True)
            return text

    def _clean_characters_in_word(self, token: str) -> str:
        """Limpieza de caracteres específicos en palabras individuales."""
        if not token:
            return token
        
        # Esta sección de reemplazos OCR confusos está desactivada por defecto.
        # Activarla puede ser útil pero requiere pruebas cuidadosas para no
        # corromper datos válidos (ej: 'S' vs '5').
        char_replacements = {
            # '0': 'O', '1': 'l', '5': 'S', '8': 'B', '|': 'I',
            # '!': '1', 'G': '6', 'g': '9', 'Z': '2', 'z': '2',
        }
        
        for wrong_char, correct_char in char_replacements.items():
            token = token.replace(wrong_char, correct_char)
        
        token = re.sub(r'[^\w\s\.,$€£¥¢]', '', token)
        
        return token

    def _normalize_char_for_freq(self, ch: str, manager: DataFormatter) -> str:
        # Mantén tildes/ñ si existen en la tabla; si no, haz fallback a su base
        if ch in self._get_frecuency_norm(manager):
            return ch
        base_map = {
            "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
            "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U",
            "ü": "u", "Ü": "U",
        }
        return base_map.get(ch, ch)

    def _get_frecuency_norm(self, manager: DataFormatter) -> Optional[Dict[str, float]]:
        try:
            if not manager or not getattr(manager, "workflow", None):
                logger.warning("Manager o workflow ausente")
                return None

            frecuency_char: Dict[str, int] = manager.get_frecuency_char()
            max_val = float(max(frecuency_char.values()))

            freq_norm: Dict[str, float] = {char: (val / max_val) * 100 for char, val in frecuency_char.items()}
            return freq_norm
        except Exception as e:
            logger.warning(f"Error al obtener frecuencias normalizadas: {e}", exc_info=True)

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

    def _save_json(self, context: Dict[str, Any], final_polygons_dict: List[Optional[Dict[str, Any]]], file_name: str):
        from services.output_service import save_json
        import os

        output_paths = context.get("output_paths", [])
        for path in output_paths:
            output_dir: str = os.path.join(path, "clean text")
            json_file_name = f"{os.path.splitext(file_name)[0]}.json"
            save_json(final_polygons_dict, output_dir, json_file_name)
        
        if output_paths:
            logger.debug(f"OCR Raw results para '{file_name}' guardado en {len(output_paths)} ubicaciones.")