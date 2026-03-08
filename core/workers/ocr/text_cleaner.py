# PerfectOCR/core/workers/ocr/text_cleaner.py
import logging
import dataclasses
import numpy as np
from typing import Dict, Any, List
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons
from core.factory.abstract_worker import OCRAbstractWorker
from core.utils.text_utils import validate_text, validate_alone_chars, space_removal, detect_punt, remove_special_sequences
from core.utils.math_utils import text_encode
from core.utils.data_utils import CHAR_FRECUENCY

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
        worker_config = config.get("text_cleaner", {})
        self.min_confidence: float  = worker_config.get("min_confidence")
        self.min_char = int(worker_config.get("min_char"))
        self.min_probability = float(worker_config.get("min_probability"))
                    
    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        logger.debug(f"Inicia cleanner")
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
            text = space_removal(text)

            if not validate_text(text):
                logger.debug(f"Eliminado {poly_id} sin texto inicial")
                eliminated_count += 1
                continue

            elif not validate_alone_chars(text):
                # logger.info(f"Eliminado {poly_id} por soledad: '{text}'")
                eliminated_count += 1
                continue

            text_sec = remove_special_sequences(text)
            if text_sec != text:
                logger.info(f"Secuencia eliminada: '{text.removeprefix(text_sec)}' en '{text}' in {polygon.polygon_id if polygon else ''}")
                continue
            
            fil_text = self.filter_low_prob_tokens(text, polygon, manager)
            if not validate_text(fil_text):
                logger.info(f"Eliminado {poly_id} sin texto después de filtrado de probabilidad")
                eliminated_count += 1
                continue

            # text = remove_special_chars(text)

            is_numeric_like = (isinstance(sc, list) and any(c in [1, 2, -2] for c in sc)) or (isinstance(sc, int) and sc in [1, 2, -2])

            if (not validate_text(text) or
                (confidence < self.min_confidence and not is_numeric_like) or detect_punt(text)):
                reason = "sin texto" if not validate_text(text) else f"baja confianza ({confidence:.2f})" if confidence < self.min_confidence and not is_numeric_like else "solo caracteres de puntuación"
                logger.info(f"Eliminado {poly_id}: '{text}' (Razón: {reason})")
                eliminated_count += 1
                continue

            # 3. Ruta Normal: Limpiar texto del polígono vacío
            cleaned_text = self.process_single_text(text, polygon)
            if validate_text(cleaned_text):
                updated_polygon = dataclasses.replace(polygon, ocr_text=cleaned_text)
                list_of_final_polygons.append(updated_polygon)
                
            else:
                logger.info(f"Eliminado {poly_id}: Sin texto en limpieza final")
                eliminated_count += 1

        eliminated_count += 1

        # 4. Reconstrucción y reindexación final
        final_polygons_dict: Dict[str, Polygons] = {}
        for idx, poly_obj in enumerate(list_of_final_polygons):
            new_id = f"poly_{idx:04d}"
            poly_index = idx
            final_poly_obj = dataclasses.replace(poly_obj, polygon_id=new_id, poly_index=poly_index)
            final_polygons_dict[new_id] = final_poly_obj
            
            # 5. Reemplazo directo en el manager
        manager.workflow.polygons = final_polygons_dict

        logger.debug(f"{eliminated_count} limpios. Total final: {len(final_polygons_dict)}")
            
        return True

    def process_single_text(self, text: str, polygon: Polygons) -> str:
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
            if not validate_alone_chars(token):
                logger.info(f"Eliminado único: '{token}' in {polygon.polygon_id if polygon else ''}")
                continue
            else:
                processed_words.append(token)
        
        return ' '.join(processed_words)

    def filter_low_prob_tokens(self, text: str, polygon: Polygons, manager: DataFormatter) -> str:
        if polygon.ocr_confidence and polygon.ocr_confidence > self.min_confidence:
            return text

        if text.isdigit() or text.isalpha():
            return text
            
        try:
            # sc = polygon.semantic_clasification
            #is_numeric_like = (isinstance(sc, list) and any(c in [1, 2, -2] for c in sc)) or \
                     #       (isinstance(sc, int) and sc in [1, 2, -2])
            #if is_numeric_like:
             #   return text

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

                #if any(ch in self.char_num for ch in t):
                 #   kept.append(tok)
                  #  continue

                eff_len = len(''.join(ch for ch in t if not ch.isspace()))
                if eff_len < self.min_char:
                    score = self.token_freq_score(t, manager)
                    
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

    def get_frecuency_norm(self, manager: DataFormatter) -> Dict[str, float]:
        try:
            max_val = max(CHAR_FRECUENCY.values())
            freq_norm: Dict[str, float] = {char: (val / max_val) * 100 for char, val in CHAR_FRECUENCY.items()}
            return freq_norm
        
        except Exception as e:
            logger.error(f"Error al obtener frecuencias normalizadas: {e}", exc_info=True)
            return {}

    def token_freq_score(self, token: str) -> float:
        """
        Calcula la probabilidad del token usando text_encode para obtener 
        la media de frecuencia (frecuencia normalizada).
        """
        if not token:
            return 100.0
            
        # Usamos text_encode con el tipo 'frequency' (que usa CHAR_FRECUENCY)
        # Esto devuelve un array de promedios por cada caracter
        encoded = text_encode(token, ["frequency"])
        
        # Obtenemos el promedio del token (axis 1) y luego el promedio general
        if encoded.size == 0:
            return 100.0
            
        freq_mean = np.mean(encoded).astype(np.float32)
        
        return float(freq_mean)
