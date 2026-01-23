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
                    new_headers = self.divide_headers(index_word, headers)
                    logger.info(f"New: {new_headers}")
                    
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
    
    def divide_headers(self, mapped_text: List[List[Any]], headers: List[str]) -> List[Dict[str, Any]]:
        current_pos = 0
        header_slices: List[Dict[str, Any]] = []
        
        # 1. Extraer slices de headers en orden
        for header in headers:
            found_indices = []
            for char in header:
                for i in range(current_pos, len(mapped_text)):
                    if mapped_text[i][3] and mapped_text[i][1] == char:
                        mapped_text[i][3] = False  # Marcar como usado
                        found_indices.append(int(mapped_text[i][0]))
                        current_pos = i + 1
                        break
            
            if found_indices:
                header_slices.append({
                    'type': 'header',
                    'content': header,
                    'start_idx': min(found_indices),
                    'end_idx': max(found_indices) + 1
                })

        # 2. Recolectar remanentes barriendo el mapeo
        final_segments: List[Dict[str, Any]] = []
        temp_rem_indices: List[int] = []
        
        for i in range(len(mapped_text)):
            idx_orig = int(mapped_text[i][0])
            is_available = mapped_text[i][3]

            if is_available:
                temp_rem_indices.append(idx_orig)
            else:
                # Si encontramos un bloque ocupado (header), cerramos remanente previo
                if temp_rem_indices:
                    final_segments.append({
                        'type': 'remainder',
                        'start_idx': min(temp_rem_indices),
                        'end_idx': max(temp_rem_indices) + 1
                    })
                    temp_rem_indices = []
                
                # Añadir el header que corresponde a este índice (si es su inicio)
                for h in header_slices:
                    if h['start_idx'] == idx_orig:
                        final_segments.append(h)
                        break

        # Cerrar remanente final si quedó algo al final de la cadena
        if temp_rem_indices:
            final_segments.append({
                'type': 'remainder',
                'start_idx': min(temp_rem_indices),
                'end_idx': max(temp_rem_indices) + 1
            })

        return final_segments