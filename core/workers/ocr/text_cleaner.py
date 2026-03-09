# PerfectOCR/core/workers/ocr/text_cleaner.py
import logging
import dataclasses
import numpy as np
import time
from typing import Dict, Any, List
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons
from core.factory.abstract_worker import OCRAbstractWorker
from core.utils.text_utils import validate_text, validate_alone_chars, space_removal, remove_special_sequences
from core.utils.math_utils import text_encode

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
        t0 = time.perf_counter()
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
            # confidence = polygon.ocr_confidence or 0.0

            text = polygon.ocr_text or ""
            text = space_removal(text)

            if not text:
                logger.debug(f"Eliminado {poly_id} sin texto inicial")
                eliminated_count += 1
                continue

            text_sec = remove_special_sequences(text)
            if text_sec != text:
                removed_chars = sorted(set(text) - set(text_sec))
                logger.debug(
                    "Secuencia especial eliminada: [%s] in %s",
                    "".join(removed_chars),
                    polygon.polygon_id if polygon else ""
                )
                eliminated_count += 1
                continue
            
            fil_text = self.filter_low_prob_tokens(text_sec, polygon)
            if not validate_text(fil_text):
                # logger.info(f"Eliminado {poly_id} sin texto después de filtrado de probabilidad")
                eliminated_count += 1
                continue

            cleaned_text = self.process_single_text(fil_text, polygon)
            if cleaned_text:
                updated_polygon = dataclasses.replace(polygon, ocr_text=cleaned_text)
                list_of_final_polygons.append(updated_polygon)
                
            else:
                # logger.info(f"Eliminado {poly_id}: Sin texto en limpieza final")
                eliminated_count += 1

        # eliminated_count += 1

        # 4. Reconstrucción y reindexación final
        final_polygons_dict: Dict[str, Polygons] = {}
        for idx, poly_obj in enumerate(list_of_final_polygons):
            new_id = f"poly_{idx:04d}"
            poly_index = idx
            final_poly_obj = dataclasses.replace(poly_obj, polygon_id=new_id, poly_index=poly_index)
            final_polygons_dict[new_id] = final_poly_obj
            
            # 5. Reemplazo directo en el manager
        manager.workflow.polygons = final_polygons_dict

        # logger.info(f"{eliminated_count} polígonos limpios en {time.perf_counter() - t0:.6f}'s")
            
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
            
            # Eliminar tokens que sean un carácter especial especificado (ej. ")")
            if not validate_alone_chars(token):
                # logger.info(f"Eliminado único: '{token}' in {polygon.polygon_id if polygon else ''}")
                continue
            else:
                processed_words.append(token)
        
        return ' '.join(processed_words)

    def filter_low_prob_tokens(self, text: str, polygon: Polygons) -> str:
        """
        Filtra por probabilidad usando el texto completo (no por token).
        Retorna "" si el texto completo se considera ruido.
        """
        try:
            sc: List[int] | int = polygon.semantic_clasification or [0]
            if not isinstance(sc, list):
                sc = [sc]

            sc_array = np.array(sc, np.int8)
            sc_real = np.mean(sc_array).astype(np.int8)
            if int(sc_real) in (1, 2, -2):
                return text

            score = self.token_freq_score(text.lower())
            if score < self.min_probability:
                logger.info(f"Eliminado:{polygon.polygon_id} | Texto:'{text}' | Probabilidad global: {score:.4f}")
                return ""

            return text

        except Exception as e:
            logger.error(f"Error eliminando texto por frecuencia global: {e}", exc_info=True)
            return text

    def token_freq_score(self, token: str) -> float:
        """
        Calcula la probabilidad del token usando text_encode para obtener 
        la media de frecuencia (frecuencia normalizada).
        """
        if not token:
            return 100.0

        encoded = text_encode(token, ["frequency"])
        return float(np.mean(encoded))

