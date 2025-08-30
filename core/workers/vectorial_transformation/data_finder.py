from data.scripts.word_finder import WordFinder
import os
import time
from typing import Dict, Any, List
import logging
from core.domain.data_models import Polygons, AllLines
from core.factory.abstract_worker import VectorizationAbstractWorker
from core.domain.data_formatter import DataFormatter

logger = logging.getLogger(__name__)

class DataFinder(VectorizationAbstractWorker):
    def __init__(self, config: Dict[str, Any], project_root: str):
        super().__init__(config, project_root)
        self.project_root = project_root
        self.worker_config = config.get('data_finder', {})

    def vectorize(self, context: Dict[str, Any], manager: DataFormatter) -> bool:
        try:
            start_time = time.time()
            logger.debug("Word_finderiniciado")
            
            if not manager or not getattr(manager, "workflow", None):
                logger.warning("Manager o workflow ausente")
                return False
            
            workflow = manager.workflow
            polygons: Dict[str, Polygons] = getattr(workflow, "polygons", {}) or {}
            all_lines: Dict[str, AllLines] = getattr(workflow, "all_lines", {}) or {}
            
            if not all_lines:
                return False
                
            if not polygons:
                logger.info("No hay polygons para procesar")
                return False
            
            # Obtener IDs de líneas para el análisis
            line_ids = list(all_lines.keys())
            if not line_ids:
                logger.info("No hay líneas para analizar")
                return False
            
            # Llamar al método original que funciona
            results = self._find_data(manager, line_ids, polygons)
            header_indices = results["header_line_indices"]
            
            # Marcar SOLO las líneas de encabezado
            for idx in header_indices:
                if idx < len(line_ids):
                    line_id = line_ids[idx]
                    if line_id in workflow.all_lines:
                        line_obj = workflow.all_lines[line_id]
                        line_obj.header_line = True
                        
                        logger.info(f"Encabezado marcado: line_id={line_id}, texto='{getattr(line_obj, 'text', '')}'")
            
            # Guardar TODA la información en el contexto
            context.update(results)
            
            # Actualiza las líneas marcadas como encabezado en las dataclasses
            updates = [(line_ids[i], {"header_line": True}) for i in header_indices if i < len(line_ids)]
            if updates:
                manager.update_lines_metadata(updates)
            # Guardar resultados en el contexto
            context['header_line_indices'] = header_indices
            context['header_line_ids'] = [line_ids[i] for i in header_indices if i < len(line_ids)]
            total_time = time.time() - start_time

            logger.info(f"Encabezados detectados (por palabra): {context['header_line_ids']} en {total_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Error detectando encabezados por palabra: {e}", exc_info=True)
            return False

    def _find_data(self, manager: DataFormatter, line_ids: List[str], polygons: Dict[str, Polygons]) -> Dict[str, Any]:
        """
        Busca encabezados por palabra usando polygons:
        - Encuentra todas las líneas con palabras de encabezado
        - Selecciona la línea más arriba (menor índice)
        - retorna lista con solo esa línea
        """
        all_matches: List[Dict[str, Any]] = []
        lines_with_headers: set[str] = set()
        lines_with_keys: Dict[str, List[Dict[str, Any]]] = {}
        lines_with_categories: Dict[str, List[Dict[str, Any]]] = {}
    
        for pid, poly in polygons.items():
            if not manager or not getattr(manager, "workflow", None):
                logger.info("_find_headers: manager o workflow ausente, saliendo")
                return []

        # ruta al modelo configurable
        model_path = None
        try:
            model_path = self.worker_config.get("wordfinder_model_path") or self.config.get("wordfinder_model_path")
        except Exception:
            model_path = None
        if not model_path:
            model_path = os.path.join(self.project_root or ".", "data", "wordfinder_model.pkl")
        logger.debug(f"_find_headers: ruta modelo WordFinder -> {model_path}")

        try:
            wf = WordFinder(model_path)
            logger.info("_find_headers: WordFinder inicializado correctamente")
        except Exception as e:
            logger.warning(f"WordFinder no pudo inicializarse con {model_path}: {e}")
            return []

        workflow = manager.workflow
        polygons: Dict[str, Polygons] = getattr(workflow, "polygons", {}) or {}
        all_lines: Dict[str, Any] = getattr(workflow, "all_lines", {}) or {}

        logger.info(f"_find_headers: cantidad polygons={len(polygons)}, cantidad all_lines={len(all_lines)}")

        # construir mapping polygon_id -> line_id
        polygon_to_line: Dict[str, str] = {}
        for lid, lobj in all_lines.items():
            for pid in getattr(lobj, "polygon_ids", []) or []:
                polygon_to_line[str(pid)] = lid
        logger.debug(f"_find_headers: mapping polygon->line construido (entradas={len(polygon_to_line)})")

        # Encontrar todas las líneas con palabras de encabezado
        lines_with_headers: set[str] = set()
        processed: int = 0
        matched: int = 0
        
        for pid, poly in polygons.items():
            processed += 1
            # obtener texto del polygon (palabra individual)
            text = ""
            try:
                if hasattr(poly, "ocr_text"):
                    text = str(poly.ocr_text or "")
                else:
                    text = str(poly.get("ocr_text", "") if isinstance(poly, dict) else "")
            except Exception:
                text = ""
                
            if not text:
                continue

            try:    
                matches: List[Dict[str, Any]] = wf.find_keywords(text)
            
                if matches:
                    matched += 1
                    line_id = polygon_to_line.get(str(pid))
                    if line_id:
                        for match in matches:
                            # Crear una copia antes de agregar contexto
                            match_with_context = match.copy()
                            match_with_context.update({
                                "polygon_id": pid,
                                "line_id": line_id,
                                "original_text": text
                            })
                            all_matches.append(match_with_context)
                            
                            # Clasificar por tipo usando la copia
                            if match_with_context.get("key_field"):
                                lines_with_headers.add(line_id)
                                if line_id not in lines_with_keys:
                                    lines_with_keys[line_id] = []
                                lines_with_keys[line_id].append(match_with_context)
                                
                            if match_with_context.get("header_category"):
                                if line_id not in lines_with_categories:
                                    lines_with_categories[line_id] = []
                                lines_with_categories[line_id].append(match_with_context)
                    
                    logger.info(f"MATCH: polygon={pid}, line_id={line_id}, text='{text}', matches={len(matches)}")
                        
            except Exception as e:
                logger.exception(f"_find_headers: WordFinder error con polygon {pid}: {e}", exc_info=True)
                
                info = wf.get_model_info()
                logger.info(f"{info}")
            # Retornar TODO
            workflow = manager.workflow
            all_lines = getattr(workflow, "all_lines", {}) or {}

        # Calcular candidatos 0-based y elegir la línea más arriba (menor índice)
        header_indices_0based = sorted({i for i, lid in enumerate(line_ids) if lid in lines_with_headers})
        if header_indices_0based:
            chosen_idx_0 = min(header_indices_0based)
            chosen_indices_0based = [chosen_idx_0]
            chosen_ids = [line_ids[chosen_idx_0]]
        else:
            chosen_indices_0based = []
            chosen_ids = []

        # Convertir a 1-based para devolver y para logs (UNIFICADO: todo 1-based hacia afuera)
        header_line_positions = [i + 1 for i in chosen_indices_0based]   # 1-based
        header_line_ids = chosen_ids
        header_line_map = {lid: (idx + 1) for lid, idx in zip(header_line_ids, chosen_indices_0based)}  # line_id -> 1-based

        # Log detallado con posición 1-based, id y texto de línea
        detected = []
        for pos1 in header_line_positions:
            idx0 = pos1 - 1
            lid = line_ids[idx0]
            line_obj = all_lines.get(lid, {})
            text_val = getattr(line_obj, "text", None) if not isinstance(line_obj, dict) else line_obj.get("text", "")
            if text_val is None:
                text_val = line_obj if isinstance(line_obj, str) else ""
            detected.append({"position": pos1, "line_id": lid, "text": str(text_val)})

        logger.info(f"Encabezado elegido (por palabra): {detected} | posiciones_1based={header_line_positions} | ids={header_line_ids}")

        # Retornar estructura completa usando 1-based como contrato externo
        return {
            "matches": all_matches,
            "lines_with_keys": lines_with_keys,
            "lines_with_categories": lines_with_categories,
            "header_line_positions": header_line_positions, # 1-based (usar en todo el pipeline)
            "header_line_ids": header_line_ids,
            "header_line_map": header_line_map  # { line_id: index_1based }
        }
