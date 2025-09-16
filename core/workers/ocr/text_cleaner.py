# PerfectOCR/core/workers/ocr/text_cleaner.py
import logging
import re
import numpy as np
import dataclasses
from typing import Dict, Any, List, Optional, Tuple
from cleantext import clean # type: ignore
from core.domain.data_formatter import DataFormatter
from core.domain.data_models import Polygons, Geometry, CroppedGeometry
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
        self.config = config.get("text_cleaner", {})
        self.enabled_outputs = self.config.get("enabled_outputs", {})
        self.output = self.config.get("clean_text", False)
        self.min_confidence_for_elimination = self.config.get("min_confidence_for_elimination", 75.0)
        self.min_contours_for_frag = self.config.get("min_contours_for_frag", 2)
                    
    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        if not manager.workflow or not manager.workflow.polygons:
            logger.warning("TextCleaner: No hay polígonos en el workflow para procesar.")
            return True

        polygons_in: Dict[str, Polygons] = manager.workflow.polygons
        contours_meta: Dict[str, Any] = context.get('contours_meta', {})
        
        list_of_final_polygons: List[Polygons] = []
        eliminated_count = 0
        fragmented_count = 0

        # Ordenar polígonos por posición vertical (y luego horizontal) para un procesamiento secuencial
        sorted_poly_ids = sorted(
            polygons_in.keys(), 
            key=lambda p_id: (polygons_in[p_id].geometry.centroid[1], polygons_in[p_id].geometry.centroid[0])
        )

        for poly_id in sorted_poly_ids:
            polygon = polygons_in[poly_id]
            text = polygon.ocr_text or ""
            confidence = polygon.ocr_confidence or 0.0

            # 1. Criterio de eliminación de basura
            if (not text.strip() or
                (confidence < self.min_confidence_for_elimination and not self._is_likely_numeric_or_code(text)) or
                re.fullmatch(r'[\s\.\-_,;:]+', text)):
                eliminated_count += 1
                logger.debug(f"Eliminado (basura): ID: {poly_id} | Texto: '{text}'")
                continue

            # 2. Lógica de fragmentación
            words = text.split()
            contour_boxes = contours_meta.get(poly_id, {}).get("contour_boxes_norm", [])
            
            # Condición para fragmentar: Múltiples palabras y un número coincidente de contornos visuales
            if len(words) > 1 and len(words) == len(contour_boxes) and len(words) >= self.min_contours_for_frag:
                logger.debug(f"Fragmentando Polígono {poly_id}: '{text}' en {len(words)} partes.")
                
                new_fragments = self._create_fragments(polygon, words, contour_boxes)
                if new_fragments:
                    list_of_final_polygons.extend(new_fragments)
                    fragmented_count += 1
                else: # Si todos los fragmentos son basura, no añadir nada
                    eliminated_count +=1
            else:
                # 3. Ruta Normal: Limpiar texto del polígono sin fragmentar
                cleaned_text = self._process_single_text(text)
                if cleaned_text:
                    updated_polygon = dataclasses.replace(polygon, ocr_text=cleaned_text)
                    list_of_final_polygons.append(updated_polygon)
                else:
                    eliminated_count += 1
                    logger.debug(f"Eliminado (texto vacío post-limpieza): ID: {poly_id} | Texto: '{text}'")

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

        
        logger.info(f"Limpieza/Fragmentación: {fragmented_count} polígonos fragmentados, {eliminated_count} eliminados. Total final: {len(final_polygons_dict)}")
        return True

    def _create_fragments(self, polygon: Polygons, words: List[str], contour_boxes: List[List[float]]) -> List[Polygons]:
        """Crea y devuelve una lista de nuevos objetos Polygons para cada fragmento válido."""
        fragments = []
        try:
            crop_w = float(polygon.cropedd_geometry.poly_dims.get('poly_width'))
            crop_h = float(polygon.cropedd_geometry.poly_dims.get('poly_height'))
            orig_x_min = float(polygon.geometry.bounding_box[0])
            orig_y_min = float(polygon.geometry.bounding_box[1])

            if not all([crop_w, crop_h]):
                 logger.warning(f"Dimensiones de recorte no válidas para poly_id {polygon.polygon_id}. Saltando fragmentación.")
                 return []

            for i, word in enumerate(words):
                processed_word = self._process_single_text(word)
                if not processed_word or re.fullmatch(r'[\s\.\-_,;:]+', processed_word):
                    continue
                
                norm_box = contour_boxes[i]
                abs_x1 = float((norm_box[0] * crop_w)) + orig_x_min
                abs_y1 = float((norm_box[1] * crop_h)) + orig_y_min
                abs_x2 = float((norm_box[2] * crop_w)) + orig_x_min
                abs_y2 = float((norm_box[3] * crop_h)) + orig_y_min

                new_bbox: List[List[Tuple[float]]] = np.array([abs_x1, abs_y1, abs_x2, abs_y2])
                new_poly_coords = np.array([
                    [abs_x1, abs_y1], [abs_x2, abs_y1],
                    [abs_x2, abs_y2], [abs_x1, abs_y2]
                ])
                new_centroid = np.mean(new_poly_coords, axis=0)

                new_geometry = Geometry(
                    polygon_coords=new_poly_coords,
                    bounding_box=new_bbox,
                    centroid=new_centroid
                )
                
                new_fragment = Polygons(
                    polygon_id="", # Se asignará en la reindexación final
                    geometry=new_geometry,
                    ocr_text=processed_word,
                    ocr_confidence=polygon.ocr_confidence,
                    was_fragmented=True,
                    line_id=polygon.line_id,
                    cropedd_geometry=CroppedGeometry(padd_centroid=np.array([]), padding_coords=np.array([]), poly_dims={}),
                    cropped_img=None,
                    perimeter=None,
                    status=True,
                    stage="TextCleaner",
                    key_field=polygon.key_field,
                    semantic_type=polygon.semantic_type
                )
                fragments.append(new_fragment)
        except Exception as e:
            logger.error(f"Error creando fragmentos para el polígono {polygon.polygon_id}: {e}", exc_info=True)
            return [] # Devuelve lista vacía en caso de error
        return fragments

    def _process_single_text(self, text: str) -> str:
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
            
            if self._is_likely_numeric_or_code(cleaned_token):
                # --- RUTA DE ALTA SEGURIDAD PARA NÚMEROS ---
                safe_word = self._safe_normalize_numeric_separators(cleaned_token)
                processed_words.append(safe_word)
            else:
                # --- RUTA NORMAL PARA TEXTO ---
                try:
                    # Se utiliza la librería clean-text para una limpieza general
                    cleaned_token_lib = clean(
                        cleaned_token, # Usar el token ya pre-limpiado
                        clean_all=False, extra_spaces=True, stemming=False,
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
        
    def _is_likely_numeric_or_code(self, token: str) -> bool:
        """
        Determina si un token es probablemente un número, moneda o código.
        Es muy inclusivo para evitar la pérdida de datos.
        """
        if not token:
            return False
        # Si contiene al menos un dígito, es candidato
        if re.search(r'\d', token):
            return True
        # Símbolos monetarios comunes
        if any(c in token for c in ['$', '€', '£']):
            return True
        # Patrones que contienen números y separadores comunes
        monetary_patterns = [
            r'^\$?\d{1,3}(,\d{3})*(\.\d+)?$', # $1,234.56
            r'^\d+[\.,]\d{2}$', # 123.45 o 123,45
            r'^\d+(\.\d{3})+(,\d+)?$', # 1.234,56
        ]
        for pattern in monetary_patterns:
            if re.match(pattern, token):
                return True
        return False

    def _safe_normalize_numeric_separators(self, token: str) -> str:
        """
        Normaliza DE FORMA SEGURA los separadores en un token numérico.
        Convierte comas a puntos solo si están entre dígitos.
        """
        # Reemplaza comas por puntos solo cuando están entre dos dígitos para decimales
        safe_word = re.sub(r'(\d),(\d)', r'\1.\2', token)
        # Elimina caracteres no numéricos excepto el punto decimal
        safe_word = re.sub(r'[^\d\.]', '', safe_word)
        return safe_word

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
        
        # Limpiar caracteres de ruido OCR que no son letras, números o puntuación común.
        # Se mantiene: letras, números, espacios, punto, coma, guion, arroba, etc.
        token = re.sub(r'[^\w\s\.,\-_@$€£]', '', token)
        
        return token

    def _save_json(self, context: Dict[str, Any], final_polygons_dict: List[Optional[Dict[str, Any]]], file_name: str):
        from services.output_service import save_json
        import os

        output_paths = context.get("output_paths", [])
        for path in output_paths:
            output_dir: str = os.path.join(path, "clean text")
            json_file_name = f"{os.path.splitext(file_name)[0]}.json"
            save_json(final_polygons_dict, output_dir, json_file_name)
        
        if output_paths:
            logger.info(f"OCR Raw results para '{file_name}' guardado en {len(output_paths)} ubicaciones.")