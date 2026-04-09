# PerfectOCR/core/workers/ocr/text_cleaner.py
import logging
import dataclasses
#port numpy as np
from typing import Dict, Any, List
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons
from core.factory.abstract_worker import OCRAbstractWorker
from core.utils.text_utils import space_removal, remove_special_sequences, validate_text, separate_punt, clean_punct
#ore.utils.math_utils import text_encode

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
        # t0 = time.perf_counter()
        if not manager.workflow or not manager.workflow.polygons:
            logger.warning("TextCleaner: No hay polígonos en el workflow para procesar.")
            return False

        polygons_in: Dict[str, Polygons] = manager.workflow.polygons

        logger.debug(f"Cantidad de polígonos recibidos:{len(polygons_in)}")
        list_of_final_polygons: List[Polygons] = []
        eliminated_count = 0

        for poly_id, polygon in polygons_in.items():
            text = polygon.ocr_text or ""
            text = text.strip()
            kf = polygon.key_field
            
            if kf or kf is not None:
                logger.debug(f"'{poly_id}' con KEYFIELD ya no se limpia: '{text_sec}'")
                updated_polygon = dataclasses.replace(polygon, ocr_text=text)
                list_of_final_polygons.append(updated_polygon)
                continue
                
            if not text or not validate_text(text):
                logger.debug(f"Eliminado {poly_id} sin texto valido incial: {text}")
                eliminated_count += 1
                continue

            txt = space_removal(text)
            if txt != text:
               logger.debug(f"Espacios eliminados de {poly_id}: '{text}' -> '{txt}'")
               
            if not txt or not validate_text(txt):
                logger.debug(f"{poly_id} sin texto válido después de espacios: {txt}")
                eliminated_count += 1
                continue

            text_sec = remove_special_sequences(txt)
            if text_sec != txt:
               sec = txt.replace(text_sec, "")
               logger.debug(f" {poly_id} Secuencia especial eliminada: '{sec}' | '{txt}' -> '{text_sec}'")
               
            if not text_sec or not validate_text(text_sec):
                logger.debug(f"{poly_id} sin texto valido después de secuencias especiales: {text_sec}")
                eliminated_count += 1
                continue
                
            fil_text = separate_punt(text_sec)
            if fil_text != text_sec:
               logger.debug(f"Separación {poly_id}: '{text_sec}' -> '{fil_text}'")

            if not fil_text or not validate_text(fil_text):
                # logger.debug(f"{poly_id} sin texto valido después de separar puntuación: {fil_text}")
                eliminated_count += 1
                continue

            cleaned_text = self.process_single_text(fil_text, polygon)
            if not cleaned_text or not validate_text(cleaned_text):
                # logger.debug(f"Eliminado {poly_id}: Sin texto en limpieza final")
                eliminated_count += 1
                continue
            
            updated_polygon = dataclasses.replace(polygon, ocr_text=cleaned_text)
            list_of_final_polygons.append(updated_polygon)
                
        # 4. Reconstrucción y reindexación final
        final_polygons_dict: Dict[str, Polygons] = {}
        for idx, poly_obj in enumerate(list_of_final_polygons):
            new_id = f"poly_{idx:04d}"
            poly_index = idx
            final_poly_obj = dataclasses.replace(poly_obj, polygon_id=new_id, poly_index=poly_index)
            final_polygons_dict[new_id] = final_poly_obj
            
            # 5. Reemplazo directo en el manager
        manager.workflow.polygons = final_polygons_dict

        # logger.debug(f"{eliminated_count} polígonos limpios en {time.perf_counter() - t0:.6f}'s")
            
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

            clean_token = clean_punct(token)
                # Eliminar tokens que sean un carácter especial especificado (ej. ")")
            if not clean_token or not validate_text(clean_token):
                # logger.debug(f"Eliminado texto basura : '{clean_token}' in {polygon.polygon_id if polygon else ' '}")
                continue
            else:
                processed_words.append(clean_token)
        
        return ' '.join(processed_words).strip()

    # def filter_low_prob_tokens(self, text: str, polygon: Polygons) -> str:
    #     """
    #     Filtra por probabilidad usando el texto completo (no por token).
    #     Retorna "" si el texto completo se considera ruido.
    #     """
    #     try:
    #         sc: List[int] = polygon.semantic_clasification or [0]

    #         sc_array = np.array(sc, np.int8)
    #         sc_real = np.median(sc_array)
    #         if int(sc_real) in (1.0, 2.0, -2.0):
    #             return text

    #         score = self.token_freq_score(text.lower())
    #         if score < self.min_probability:
    #             # logger.debug(f"Eliminado:{polygon.polygon_id} | Texto:'{text}' | Probabilidad global: {score:.4f}")
    #             return ""

    #         return text

    #     except Exception as e:
    #         logger.error(f"Error eliminando texto por frecuencia global: {e}", exc_info=True)
    #         return text

    # def token_freq_score(self, token: str) -> float:
    #     """
    #     Calcula la probabilidad del token usando text_encode para obtener 
    #     la media de frecuencia (frecuencia normalizada).
    #     """
    #     if not token:
    #         return 100.0

    #     #encoded = text_encode(token, ["frequency"])
    #     #return float 
    #     return h
