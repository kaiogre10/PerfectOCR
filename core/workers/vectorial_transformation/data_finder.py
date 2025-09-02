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
            logger.debug("Scanner iniciado")
            
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
            header_indices = self._find_data(manager, line_ids)
            
            # Marcar las líneas como encabezados en las dataclasses
            for idx in header_indices:
                if idx < len(line_ids):
                    line_id = line_ids[idx]
                    if line_id in workflow.all_lines:
                        # Marcar la línea como encabezado
                        line_obj = workflow.all_lines[line_id]
                        line_obj.header_line = True
            
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
            logger.error(f"Error detectando encabezados por palabra: {e}")
            return False

    def _find_data(self, manager: DataFormatter, line_ids: List[str]) -> List[int]:
        """
        Busca encabezados por palabra usando polygons:
        - Encuentra todas las líneas con palabras de encabezado
        - Selecciona la línea más arriba (menor índice)
        - retorna lista con solo esa línea
        """
        logger.debug("_find_headers: inicio de búsqueda de encabezados")
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
                results: List[Dict[str, Any]] = wf.find_keywords(text)
                
                if results:
                    matched += 1
                    line_id = polygon_to_line.get(str(pid))
                    if line_id:
                        lines_with_headers.add(line_id)
                        logger.info(f"MATCH: polygon={pid}, line_id={line_id}, text='{text}'")
                        
            except Exception as e:
                logger.exception(f"_find_headers: WordFinder error con polygon {pid}: {e}", exc_info=True)

        logger.debug(f"_find_headers: processed={processed}, matches={results}, lines_with_headers={len(lines_with_headers)}")

        if not lines_with_headers:
            logger.info("_find_headers: no se encontraron encabezados, retornando []")
            return []
        
        # model_info = wf.get_model_info()
        # logger.info(f"{model_info}")
        
        # Seleccionar solo la línea más arriba (menor índice)
        header_line_id = min(lines_with_headers, key=lambda x: int(x.split('_')[1]))
        header_index = line_ids.index(header_line_id)
        
        logger.info(f"_find_headers: línea más arriba seleccionada: {header_line_id} (índice {header_index})")
        return [header_index]