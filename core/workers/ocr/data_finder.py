# core/workers/ocr/data_finder.py
import time
from typing import Dict, Any, Optional, List
import logging
from core.domain.data_models import Polygons
from core.factory.abstract_worker import OCRAbstractWorker
from core.domain.data_formatter import DataFormatter
from core.domain.models_manager import ModelsManager
from core.utils.text_validator import validate_text, norm_text
from core.utils.pattern_finder import find_rfc, find_iva, find_date

logger = logging.getLogger(__name__)

class DataFinder(OCRAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get('data_finder', {})
        self.max_q_lenght = self.worker_config["max_q_lenght"]
        self._model = None

    @property
    def model(self) -> Optional[Any]:
        try:
            if self._model is None: #type: ignore
                model_manager = ModelsManager.get_instance()
                self._model = model_manager.word_finder #type: ignore
                logger.warning("Modelo de búsqueda obtenido del ModelsManager")
            return self._model #type: ignore

        except Exception as e:
            logger.error(f"DataFinder: Modelo de búsqueda no disponible en ModelManager{e}", exc_info=True)
            return None

    def transcribe(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        logger.debug("Data Finder iniciado")
        start_time = time.perf_counter()
        try:
            polygons: Dict[str, Polygons] = manager.workflow.polygons if manager.workflow else {}
            if not polygons:
                logger.error("No hay polygons para procesar")
                return False
            
            # Llamar al meetodo original que funciona
            polygon_updates = self._find_data(polygons)

            # Actualiza los key_fields
            if manager.update_key_field(polygon_updates):
                logger.info(f"Key Fields detectados en {time.perf_counter() - start_time:.6f}s")
                return True
                
        except Exception as e:
            logger.error(f"Error detectando encabezados por palabra: {e}", exc_info=True)
        return True

    def _find_data(self, polygons: Dict[str, Polygons]) -> Dict[str, int]:
        time0 = time.perf_counter()
        if self.model is None:
            logger.error("DataFinder no iniciado, no se puede búsacar texto")
            return {}

        try:
            processed_count = 0
            polygon_updates: Dict[str, int] = {}
            skipped_semantic = 0
            skipped_len = 0

            for pid, poly in polygons.items():
                processed_count += 1

                sc = poly.semantic_clasification
                if sc == 1 or sc == 2 or sc == -1 or sc == -2:
                    logger.debug(f"{pid} omitido semanticamente sc= '{sc}'")
                    skipped_semantic += 1
                    continue

                ocr_text = poly.ocr_text or ""
                word_lenght = len(ocr_text)
                if not validate_text(ocr_text) or word_lenght < self.max_q_lenght[0] or word_lenght > self.max_q_lenght[1]:
                    logger.debug(f"{pid} sin texto o excede longitud: '{ocr_text}', letras: '{word_lenght}'")
                    skipped_len += 1
                    continue

                date_key = find_date(ocr_text)
                if date_key:
                    skipped_semantic +=1
                    logger.warning(f"FECHA encontrado en {pid}, '{ocr_text}'")
                    polygon_updates[pid] = 9
                    continue

                rfc_key = find_rfc(ocr_text)
                if rfc_key:
                    skipped_semantic +=1
                    logger.warning(f"RFC encontrado en {pid}, '{ocr_text}'")
                    polygon_updates[pid] = 7
                    continue

                iva_key = find_iva(ocr_text)
                if iva_key:
                    skipped_semantic +=1
                    logger.warning(f"IVA encontrado en {pid}, '{ocr_text}'")
                    polygon_updates[pid] = 8
                    continue

                valid_results: List[Dict[str, Any]] = self.model.find_keywords(ocr_text)
                if not valid_results:
                    continue

                num_keywords = len(valid_results)
                
                # logger.info(f"RESULTADO DE '{pid}': {valid_results}, cantidad de keyfields: {len(valid_results)}")
                header_kw: int = 0
                headers: List[str] = []
                for result in valid_results:
                    # best_result: Dict[str, Any] = max(valid_results, key=lambda x: x['similarity'])
                    key_field: int = result['key_field']
                    ind_header: str = result["word_found"]
                    begging: int = result["start"]
                    polygon_updates[pid] = key_field
                    header_kw += key_field
                    headers.append(ind_header)

                if num_keywords > 1 and header_kw/num_keywords == 6:
                    # Ordenar valid_results por la posición de inicio (start)
                    valid_results_sorted = sorted(valid_results, key=lambda x: x['start'])
                    
                    # Extraer los headers en orden
                    headers: List[str] = [result["word_found"] for result in valid_results_sorted]
                    
                    # logger.info(f"Poligono {pid} a segmentar: {valid_results_sorted}")
                    index_word: List[List[str]] = self.mapp_words(ocr_text)
                    # logger.info(f"Headers ordenados: {headers}")
                    self.divide_headers(index_word, headers)
                    
                    # logger.info(f"Resultado de {pid}: {result}, cantidad de keyfields: {len(valid_results)}")

            if polygon_updates:
                logger.debug(f"{skipped_semantic} polígonos semánticos omitidos")
                logger.debug(f"Encontradas {len(polygon_updates)} coincidencias en {time.perf_counter() - time0:6f}s")
                return polygon_updates

            else:
                logger.warning("No se encontraron coincidencias de palabras clave")
                return {}
                    
        except Exception as e:
            logger.warning(msg=f"Fallo en búsqueda de datos globales: {e}", exc_info=True)
            return {}
    
    def mapp_words(self, text: str) -> List[List[Any]]:
        char_mapping: List[List[Any]] = []
        
        for i, char in enumerate(text):
            # [índice_original, carácter_normalizado, header_char_asignado, disponible]
            char_mapping.append([str(i), norm_text(char), '', True])
        
        return char_mapping
    
    def divide_headers(self, mapped_text: List[List[Any]], headers: List[str]) -> None:
        current_index = 0
        header_number = 0
        header_ranges: List[Dict[str, Any]] = []
        
        for header in headers:
            header_number += 1
            start_range: Optional[int] = None  # Se asigna al encontrar la primera letra
            
            # Asignar cada letra del header (ya viene normalizado y corregido)
            for char in header:
                found = False
                
                # Buscar coincidencia exacta desde current_index en adelante
                for i in range(current_index, len(mapped_text)):
                    # Solo considerar posiciones disponibles
                    if mapped_text[i][3] and mapped_text[i][1] == char:
                        # Coincidencia exacta encontrada
                        mapped_text[i][2] = char  # Asignar carácter del header
                        mapped_text[i][3] = False  # Marcar como no disponible
                        current_index = i + 1
                        found = True
                        if start_range is None:  # Primera letra encontrada
                            start_range = i
                        break
                
                if not found:
                    # No hay coincidencia exacta, saltar al siguiente disponible sin asignar
                    for i in range(current_index, len(mapped_text)):
                        if mapped_text[i][3]:
                            # Solo marcar como no disponible, NO asignar carácter
                            mapped_text[i][3] = False
                            current_index = i + 1
                            if start_range is None:  # Primera posición usada
                                start_range = i
                            break
            
            end_range = current_index
            if start_range is None:
                start_range = end_range  # Seguridad si no encontró nada
            
            header_ranges.append({
                'header': header,
                'start': start_range,
                'end': end_range
            })
        
        # Después de asignar todos los headers, fusionar caracteres especiales residuales
        special_chars = '.,;:-_/\\|'
        for i in range(len(header_ranges)):
            # Buscar caracteres especiales ANTES del start del header (residuos a la izquierda)
            header_start = header_ranges[i]['start']
            while header_start > 0:
                prev_idx = header_start - 1
                if mapped_text[prev_idx][3]:  # Si está disponible
                    char = mapped_text[prev_idx][1]
                    if char in special_chars:
                        # Fusionar con el header actual (extender hacia la izquierda)
                        mapped_text[prev_idx][2] = char
                        mapped_text[prev_idx][3] = False
                        header_ranges[i]['start'] = prev_idx
                        header_start = prev_idx
                    else:
                        break
                else:
                    break
            
            # Buscar caracteres especiales DESPUÉS del end del header
            header_end = header_ranges[i]['end']
            while header_end < len(mapped_text):
                if mapped_text[header_end][3]:  # Si está disponible
                    char = mapped_text[header_end][1]
                    if char in special_chars:
                        # Fusionar con el header actual
                        mapped_text[header_end][2] = char
                        mapped_text[header_end][3] = False
                        header_ranges[i]['end'] = header_end + 1
                        header_end += 1
                    else:
                        break
                else:
                    break
        
        logger.info(f"Tabla final mapped_text: {mapped_text}")
        logger.info(f"Rangos de headers: {header_ranges}")
        
        # Analizar qué hay entre los headers
        divisions: List[Dict[str, Any]] = []
        
        for i in range(len(header_ranges)):
            current_header = header_ranges[i]
            
            # Calcular rangos basados en índices originales del mapped_text
            start_idx = int(mapped_text[current_header['start']][0]) if current_header['start'] < len(mapped_text) else current_header['start']
            end_idx = int(mapped_text[current_header['end'] - 1][0]) + 1 if current_header['end'] > 0 and current_header['end'] <= len(mapped_text) else current_header['end']
            
            # Texto del header usando los caracteres normalizados del mapped_text
            header_text = ''.join([mapped_text[j][1] for j in range(current_header['start'], current_header['end'])])
            
            divisions.append({
                'type': 'header',
                'header': current_header['header'],
                'start_idx': start_idx,
                'end_idx': end_idx,
                'text': header_text
            })
            
            # Ver qué hay entre este header y el siguiente
            if i < len(header_ranges) - 1:
                next_header = header_ranges[i + 1]
                between_start = current_header['end']
                between_end = next_header['start']
                
                if between_start < between_end:
                    # Calcular índices originales para el contenido intermedio
                    between_start_orig = int(mapped_text[between_start][0]) if between_start < len(mapped_text) else between_start
                    between_end_orig = int(mapped_text[between_end - 1][0]) + 1 if between_end > 0 and between_end <= len(mapped_text) else between_end
                    
                    # Hay contenido entre los headers
                    between_text = ''.join([mapped_text[j][1] for j in range(between_start, between_end)])
                    between_text_stripped = between_text.strip()
                    
                    # Clasificar el contenido intermedio
                    if not between_text_stripped:
                        # Solo espacios - se elimina
                        divisions.append({
                            'type': 'spaces',
                            'action': 'delete',
                            'start_idx': between_start_orig,
                            'end_idx': between_end_orig,
                            'text': between_text
                        })
                    elif all(c in ' .,;:-_/\\|' for c in between_text_stripped):
                        # Solo caracteres especiales - asignar a la izquierda
                        divisions.append({
                            'type': 'special_chars',
                            'action': 'merge_left',
                            'start_idx': between_start_orig,
                            'end_idx': between_end_orig,
                            'text': between_text
                        })
                    else:
                        # Texto o números - crear nuevo polígono
                        divisions.append({
                            'type': 'content',
                            'action': 'new_polygon',
                            'start_idx': between_start_orig,
                            'end_idx': between_end_orig,
                            'text': between_text
                        })
        
        logger.info("=" * 80)
        logger.info("DIVISIÓN DE POLÍGONO:")
        for div in divisions:
            if div['type'] == 'header':
                logger.info(f"  HEADER '{div['header']}' [{div['start_idx']}-{div['end_idx']}]: '{div['text']}'")
            elif div['type'] == 'spaces':
                logger.info(f"  ESPACIOS [{div['start_idx']}-{div['end_idx']}]: '{div['text']}' -> ELIMINAR")
            elif div['type'] == 'special_chars':
                logger.info(f"  ESPECIALES [{div['start_idx']}-{div['end_idx']}]: '{div['text']}' -> FUSIONAR A LA IZQUIERDA")
            elif div['type'] == 'content':
                logger.info(f"  CONTENIDO [{div['start_idx']}-{div['end_idx']}]: '{div['text']}' -> NUEVO POLÍGONO")
        logger.info("=" * 80)
        
        return None