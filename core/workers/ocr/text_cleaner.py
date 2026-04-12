# PerfectOCR/core/workers/ocr/text_cleaner.py
import logging
import time
import dataclasses
from typing import Dict, Any, List
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons
from core.factory.abstract_worker import OCRAbstractWorker
from core.utils.text_utils import validate_text, remove_special_sequences, punct_strip, separate_punt

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
        self.min_probability = float(worker_config.get("min_probability"))
                    
    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        logger.debug(f"Inicia cleanner")
        t0 = time.perf_counter()
        if not manager.workflow or not manager.workflow.polygons:
            logger.warning("TextCleaner: No hay polígonos en el workflow para procesar.")
            return False

        polygons_in: Dict[str, Polygons] = manager.workflow.polygons

        logger.debug(f"Cantidad de polígonos recibidos:{len(polygons_in)}")
        list_of_final_polygons: List[Polygons] = []
        eliminated_count = 0

        for poly_id, polygon in polygons_in.items():
            kf = polygon.key_field
            if kf or kf is not None or polygon.semantic_clasification == [0]:
                logger.debug(f"'{poly_id}' con KEYFIELD ya no se limpia: '{polygon.ocr_text}'")
                updated_polygon = dataclasses.replace(polygon)
                list_of_final_polygons.append(updated_polygon)
                continue
            
            text = polygon.ocr_text or ""
            text = text.strip()
                
            if not text or not validate_text(text):
                # logger.info(f"Eliminado {poly_id} sin texto valido incial: '{text}'")
                eliminated_count += 1
                continue
            
            text_sec = remove_special_sequences(text)
            if text_sec != text:
               logger.debug(f"{poly_id} Secuencia especial eliminada: '{text}' -> '{text_sec}'")
               
            if not text_sec or not validate_text(text_sec):
                # logger.info(f"{poly_id} sin texto valido después de secuencias especiales: '{text}'")
                eliminated_count += 1
                continue

            sep_text = separate_punt(text_sec)
            if sep_text != text_sec:
                logger.debug(f"{poly_id} Texto separado: '{text_sec}' -> '{sep_text}'")

            if not sep_text or not validate_text(sep_text):
                # logger.info(f"{poly_id} sin texto válido después de eliminar puntuaciones '{text_sec}'")
                eliminated_count += 1
                continue

            txt = self.process_single_text(sep_text, polygon)
            if not txt or not validate_text(txt):
                # logger.info(f"{poly_id} sin texto válido después de espacios: '{sep_text}'")
                eliminated_count += 1
                continue
            
            if txt != text:
                logger.debug(f"Texto limpiado: '{text}' -> '{txt}'")
            
            updated_polygon = dataclasses.replace(polygon, ocr_text=txt)
            list_of_final_polygons.append(updated_polygon)
            continue
                
        # 4. Reconstrucción y reindexación final
        final_polygons_dict: Dict[str, Polygons] = {}
        for idx, poly_obj in enumerate(list_of_final_polygons):
            new_id = f"poly_{idx:04d}"
            poly_index = idx
            final_poly_obj = dataclasses.replace(poly_obj, polygon_id=new_id, poly_index=poly_index)
            final_polygons_dict[new_id] = final_poly_obj
            
            # 5. Reemplazo directo en el manager
        manager.workflow.polygons = final_polygons_dict

        logger.debug(f"Polígonos limpios en {time.perf_counter() - t0:.6f}'s")
            
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
                # logger.info(f"Eliminado texto basura : '{clean_token}' in {polygon.polygon_id if polygon else ' '}")
                continue
            else:
                processed_words.append(clean_token)
                continue
        
        return ' '.join(processed_words).strip()
